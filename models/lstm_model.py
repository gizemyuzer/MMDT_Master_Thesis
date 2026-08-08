"""
DualEncoderLSTM — Tezin "sequence baseline" modeli.

Mimari mantığı DualEncoderTransformer ile birebir aynı:
  - İki ayrı stream (technical + fundamental)
  - Her stream kendi encoder'ında sequence işleniyor
  - Pooled embedding'ler birleştirilip classifier'a giriyor

Tek fark: sequence operator olarak LSTM kullanılıyor (Transformer yerine).

Tez içindeki rolü:
  Bu model, "self-attention LSTM'den iyi mi?" sorusunu cevaplar. Eğer dual-encoder
  Transformer + cross-attention LSTM'i geçerse, hem self-attention'ın hem cross-attention
  fusion'ın katkısı kanıtlanmış olur.

Fusion modları:
  - 'concat': h_t ve h_f concat → MLP (basit baseline)
  - 'attention': h_t ve h_f arasında attention-weighted sum (orta seviye)
"""

import torch
import torch.nn as nn


class _LSTMStreamEncoder(nn.Module):
    """
    Tek bir modalite için: projection -> LSTM -> pooled embedding.

    Transformer'daki _StreamEncoder ile aynı arayüze sahip ki fair karşılaştırma
    yapılabilsin: input (B, T, input_dim), output pooled (B, d_model) embedding.
    """

    def __init__(self, input_dim, d_model, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.input_ln = nn.LayerNorm(d_model)

        # Bidirectional LSTM — hem geçmişten geleceğe hem tersi yönde bağlam kurar
        # Bu sequence learning için Transformer'a daha adil bir karşılaştırma sağlar
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,  # bidirectional → her yönde d_model/2,
            # concat sonrası toplam d_model
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        returns: (B, d_model) — pooled sequence representation
        """
        h = self.input_ln(self.proj(x))  # (B, T, D)
        lstm_out, (h_n, c_n) = self.lstm(h)  # (B, T, D), (2*n_layers, B, D/2)

        # Pooled representation: hem son timestep'in iki yönlü hidden state'i
        # hem de tüm sequence'in mean'ini kullanmak yaygın bir pratik
        # Burada sade tutuyoruz: son hidden state'i alıyoruz (her iki yönden)
        # h_n: (num_layers * 2, B, D/2) — son katmanın forward + backward'ı al
        h_forward = h_n[-2, :, :]  # (B, D/2)
        h_backward = h_n[-1, :, :]  # (B, D/2)
        pooled = torch.cat([h_forward, h_backward], dim=-1)  # (B, D)

        return self.out_ln(self.dropout(pooled))


class DualEncoderLSTM(nn.Module):
    """
    LSTM tabanlı dual-encoder — Transformer dual-encoder'ın sequence baseline'ı.

    Args:
        tech_dim: technical feature sayısı
        fund_dim: fundamental feature sayısı
        d_model: hidden dimension (her stream için, bidirectional sonrası toplam)
        n_layers: LSTM katman sayısı
        dropout: tüm projection ve LSTM'ler için dropout oranı
        fusion_type: 'concat' veya 'attention'
            - concat: [h_t, h_f] → MLP → classifier (basit)
            - attention: h_t ve h_f arasında öğrenilebilir attention weights
                          (Transformer cross-attention'ın LSTM versiyonu — fair karşılaştırma)
    """

    def __init__(
            self,
            tech_dim: int,
            fund_dim: int,
            d_model: int = 64,
            n_layers: int = 2,
            dropout: float = 0.15,
            fusion_type: str = 'concat',
    ):
        super().__init__()
        assert fusion_type in ('concat', 'attention'), \
            f"fusion_type 'concat' veya 'attention' olmalı, '{fusion_type}' verildi"

        self.tech_dim = tech_dim
        self.fund_dim = fund_dim
        self.d_model = d_model
        self.fusion_type = fusion_type

        # ── İki ayrı LSTM encoder ──────────────────────────────────
        self.tech_encoder = _LSTMStreamEncoder(
            input_dim=tech_dim, d_model=d_model,
            n_layers=n_layers, dropout=dropout
        )
        self.fund_encoder = _LSTMStreamEncoder(
            input_dim=fund_dim, d_model=d_model,
            n_layers=n_layers, dropout=dropout
        )

        # ── Attention fusion (opsiyonel — cross-attention'a alternatif) ──
        if fusion_type == 'attention':
            # Tech ve fund embedding'leri arasında bidirectional attention
            # Cross-attention'ın LSTM dünyasındaki karşılığı: query-key-value
            # mekaniği aynı, ama input'lar sequence değil pooled vektörler
            self.attn_t2f = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1),
            )
            self.attn_f2t = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1),
            )

        # ── Fusion MLP ──────────────────────────────────────────────
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # ── Classifier head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_tech, x_fund, return_embedding=False):
        """
        x_tech: (B, T, tech_dim)
        x_fund: (B, T, fund_dim)
        """
        # ── 1. Her stream'i ayrı encode et ──────────────────────────
        h_t = self.tech_encoder(x_tech)  # (B, D)
        h_f = self.fund_encoder(x_fund)  # (B, D)

        # ── 2. Fusion ───────────────────────────────────────────────
        if self.fusion_type == 'attention':
            # Tech ve fund pooled embedding'leri arasında öğrenilebilir attention
            concat_tf = torch.cat([h_t, h_f], dim=-1)
            alpha_t = torch.sigmoid(self.attn_t2f(concat_tf))  # (B, 1)
            alpha_f = torch.sigmoid(self.attn_f2t(concat_tf))  # (B, 1)

            # Weighted residual: h_t kendi içeriğini koruyor + h_f'den ağırlıklı katkı
            h_t_fused = h_t + alpha_t * h_f
            h_f_fused = h_f + alpha_f * h_t
        else:
            h_t_fused = h_t
            h_f_fused = h_f

        # ── 3. Birleştir ───────────────────────────────────────────
        fused = self.fusion_mlp(torch.cat([h_t_fused, h_f_fused], dim=-1))

        if return_embedding:
            return fused

        return self.classifier(fused)
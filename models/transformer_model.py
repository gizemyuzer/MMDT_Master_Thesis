import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# Pozisyonel Encoding (öğrenilebilir)
# ═══════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.register_buffer('positions', torch.arange(seq_len))

    def forward(self, x):
        # x: (B, T, D) — T pozisyon embedding'i ile broadcast edilir
        pos = self.pos_embedding(self.positions[: x.size(1)])
        return self.dropout(x + pos.unsqueeze(0))


# ═══════════════════════════════════════════════════════════════════
# Baseline: Tek Encoder Transformer (ablation amaçlı korundu)
# ═══════════════════════════════════════════════════════════════════
class TimeSeriesTransformer(nn.Module):
    """
    Tek-encoder Transformer baseline.
    Teknik ve fundamental özellikler tek feature vector'da birleştirilmiş şekilde gelir.
    Dual-encoder mimarisi ile karşılaştırma (ablation) için korundu.
    """

    def __init__(
            self,
            feature_dim: int,
            seq_len: int = 20,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            ffn_dim: int = 256,
            dropout: float = 0.15,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_ln = nn.LayerNorm(d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = PositionalEncoding(seq_len + 1, d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, return_embedding=False):
        # x: (B, T, feature_dim)
        h = self.input_ln(self.input_proj(x))  # (B, T, D)
        cls = self.cls_token.expand(h.size(0), -1, -1)  # (B, 1, D)
        h = torch.cat([cls, h], dim=1)  # (B, T+1, D)
        h = self.pos_enc(h)
        h = self.transformer(h)  # (B, T+1, D)

        cls_emb = h[:, 0, :]  # (B, D)

        if return_embedding:
            return cls_emb
        return self.classifier(cls_emb)


# ═══════════════════════════════════════════════════════════════════
# Dual-Encoder Transformer (TEZİN ANA MİMARİSİ)
# ═══════════════════════════════════════════════════════════════════
class _StreamEncoder(nn.Module):
    """
    Tek bir modalite için: projection -> CLS token -> positional -> Transformer.
    Output: (B, T+1, D) — CLS dahil token-seviyesi temsiller.
    """

    def __init__(self, input_dim, seq_len, d_model, n_heads, n_layers, ffn_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.input_ln = nn.LayerNorm(d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = PositionalEncoding(seq_len + 1, d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        h = self.input_ln(self.proj(x))  # (B, T, D)
        cls = self.cls_token.expand(h.size(0), -1, -1)  # (B, 1, D)
        h = torch.cat([cls, h], dim=1)  # (B, T+1, D)
        h = self.pos_enc(h)
        h = self.transformer(h)  # (B, T+1, D)
        return h


class DualEncoderTransformer(nn.Module):
    """
    Esnek encoder yapısı — üç modaliteyi destekler:
      - 'tech_only':   sadece technical encoder (price-based baseline)
      - 'fund_only':   sadece fundamental encoder (fundamental baseline)
      - 'multimodal':  her iki encoder + fusion (tezin ana modeli)

    Bu yapı tezin "modality ablation" iddiasını destekler:
    Multi-modal yaklaşımı tek-modaliteli baseline'larla karşılaştırarak
    "modaliteleri birleştirmek katma değer üretiyor mu?" sorusunu cevaplar.

    Fusion modları (sadece multimodal için anlamlı):
      - 'concat': pooled embedding'leri concatenate eder (statik füzyon, ablation)
      - 'cross_attention': çift yönlü cross-attention (dinamik füzyon)
      - 'gated_cross_attention': cross-attention + öğrenilebilir kapı
            (Zong & Zhou 2024, MSGCA mimarisine hizalı)
      - 'film': Feature-wise Linear Modulation (Perez et al. 2018). Fundamental
            CLS embedding'i, technical CLS embedding'i ölçekleyip kaydıran
            (gamma, beta) üretir. Cross-attention'dan çok daha kısıtlı bir
            mekanizma — token-seviyesi attention yok, sadece tek bir affine
            dönüşüm. Hipotez: cross-attention'ın esnekliği bu veri ölçeğinde
            overfitting'e yol açıyor olabilir; FiLM'in kısıtlılığı bunu azaltabilir.

    Füzyon merdiveni — tezin ana ablation ekseni:
        concat (statik) → cross_attention (dinamik) → gated (bağlam-duyarlı)
    FiLM bu merdivenin dışında, ayrı bir hipotez olarak değerlendirilir
    (esneklik azaltma denemesi — regularized/warmup varyantlarından farklı
    olarak mimari düzeyde kısıtlama getiriyor, hyperparameter değil).

    Gating mantığı: düz cross-attention karşı modaliteyi HER ZAMAN aynı ağırlıkla
    karıştırır. Gated versiyonda sigmoid bir kapı, "bu bağlamda karşı modaliteye
    ne kadar güveneyim?" sorusuna örnek-bazlı cevap verir. Örneğin yüksek VIX
    rejiminde fundamental sinyale ağırlık artabilir, sakin dönemde kısılabilir.
    """

    def __init__(
            self,
            tech_dim: int,
            fund_dim: int,
            seq_len: int = 20,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            ffn_dim: int = 256,
            dropout: float = 0.15,
            modality: str = 'multimodal',
            fusion_type: str = 'cross_attention',
    ):
        super().__init__()
        assert modality in ('tech_only', 'fund_only', 'multimodal'), \
            f"modality 'tech_only', 'fund_only' veya 'multimodal' olmalı, '{modality}' verildi"
        assert fusion_type in ('cross_attention', 'gated_cross_attention', 'concat', 'film'), \
            f"fusion_type 'cross_attention', 'gated_cross_attention', 'concat' veya 'film' " \
            f"olmalı, '{fusion_type}' verildi"

        self.tech_dim = tech_dim
        self.fund_dim = fund_dim
        self.d_model = d_model
        self.modality = modality
        self.fusion_type = fusion_type

        # ── Encoder'ları sadece gerekli olanları oluştur ────────────
        if modality in ('tech_only', 'multimodal'):
            self.tech_encoder = _StreamEncoder(
                input_dim=tech_dim, seq_len=seq_len,
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                ffn_dim=ffn_dim, dropout=dropout
            )
        if modality in ('fund_only', 'multimodal'):
            self.fund_encoder = _StreamEncoder(
                input_dim=fund_dim, seq_len=seq_len,
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                ffn_dim=ffn_dim, dropout=dropout
            )

        # ── Fusion sadece multimodal modda anlamlı ─────────────────
        if modality == 'multimodal':
            if fusion_type in ('cross_attention', 'gated_cross_attention'):
                # Çift yönlü cross-attention
                self.cross_t2f = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )
                self.cross_f2t = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )
                self.norm_t = nn.LayerNorm(d_model)
                self.norm_f = nn.LayerNorm(d_model)

            if fusion_type == 'gated_cross_attention':
                # Kapı projeksiyonları: [orijinal ; cross-attn çıktısı] → skaler kapı
                # Sigmoid ile [0,1] aralığına sıkıştırılır, token ve örnek bazında
                # karşı modalitenin ne kadar karışacağını belirler.
                self.gate_t = nn.Linear(d_model * 2, d_model)
                self.gate_f = nn.Linear(d_model * 2, d_model)
                # Bias'ı hafif pozitif başlat → eğitimin başında kapı yarı açık,
                # düz cross-attention davranışına yakın bir noktadan başlanır
                nn.init.zeros_(self.gate_t.weight)
                nn.init.zeros_(self.gate_f.weight)
                nn.init.constant_(self.gate_t.bias, 0.0)
                nn.init.constant_(self.gate_f.bias, 0.0)

            if fusion_type == 'film':
                # Fundamental CLS embedding → (gamma, beta) çifti üretir.
                # gamma technical embedding'i ölçekler, beta kaydırır:
                #   h_t' = gamma ⊙ h_t + beta
                # Kimlik dönüşümünden başlar (gamma=1, beta=0 init) — eğitimin
                # başında technical akışı olduğu gibi geçer, sapma öğrenilir.
                self.film_gen = nn.Linear(d_model, d_model * 2)
                nn.init.zeros_(self.film_gen.weight)
                nn.init.zeros_(self.film_gen.bias)

            # Pooled temsilleri birleştir (concat/cross_attention/gated için;
            # film kendi tek-akış çıktısını doğrudan classifier'a verir)
            if fusion_type != 'film':
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                    nn.Dropout(dropout),
                )

        # ── Classifier head (her modda d_model boyutlu input alır) ──
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_tech, x_fund, return_embedding=False):
        """
        x_tech: (B, T, tech_dim) — teknik göstergeler (fund_only modunda ignore edilir)
        x_fund: (B, T, fund_dim) — fundamental göstergeler (tech_only modunda ignore edilir)

        Trainer her iki tensoru geçirebilir, model modality'ye göre seçim yapar.
        Tek-stream modlarda diğer tensor None de olabilir.
        """
        # ── Tek modaliteli modlar — fusion yok ─────────────────────
        if self.modality == 'tech_only':
            h = self.tech_encoder(x_tech)  # (B, T+1, D)
            emb = h[:, 0, :]  # CLS token (B, D)
            if return_embedding:
                return emb
            return self.classifier(emb)

        if self.modality == 'fund_only':
            h = self.fund_encoder(x_fund)
            emb = h[:, 0, :]
            if return_embedding:
                return emb
            return self.classifier(emb)

        # ── Multimodal: her iki encoder + fusion ───────────────────
        h_t = self.tech_encoder(x_tech)  # (B, T+1, D)
        h_f = self.fund_encoder(x_fund)  # (B, T+1, D)

        if self.fusion_type == 'gated_cross_attention':
            # Cross-attention çıktıları
            attn_t, _ = self.cross_t2f(query=h_t, key=h_f, value=h_f)
            attn_f, _ = self.cross_f2t(query=h_f, key=h_t, value=h_t)

            # Öğrenilebilir kapı: karşı modalitenin katkısını modüle eder
            # g ∈ (0,1);  g→0: karşı modaliteyi yok say, g→1: tam karıştır
            g_t = torch.sigmoid(self.gate_t(torch.cat([h_t, attn_t], dim=-1)))
            g_f = torch.sigmoid(self.gate_f(torch.cat([h_f, attn_f], dim=-1)))

            h_t_fused = self.norm_t(h_t + g_t * attn_t)
            h_f_fused = self.norm_f(h_f + g_f * attn_f)

            # Kapı istatistiğini sakla — analiz/görselleştirme için
            self.last_gate_t = g_t.detach().mean().item()
            self.last_gate_f = g_f.detach().mean().item()

            cls_t = h_t_fused[:, 0, :]
            cls_f = h_f_fused[:, 0, :]
        elif self.fusion_type == 'cross_attention':
            # Tech sequence, fund sequence'i sorgular (rezidüel bağlantı)
            attn_t, _ = self.cross_t2f(query=h_t, key=h_f, value=h_f)
            h_t_fused = self.norm_t(h_t + attn_t)

            # Fund sequence, tech sequence'i sorgular (rezidüel bağlantı)
            attn_f, _ = self.cross_f2t(query=h_f, key=h_t, value=h_t)
            h_f_fused = self.norm_f(h_f + attn_f)

            cls_t = h_t_fused[:, 0, :]
            cls_f = h_f_fused[:, 0, :]
        elif self.fusion_type == 'film':
            cls_t = h_t[:, 0, :]
            cls_f = h_f[:, 0, :]

            gamma_raw, beta = self.film_gen(cls_f).chunk(2, dim=-1)
            gamma = 1.0 + gamma_raw  # kimlik dönüşümünden başla (init: gamma=1)
            cls_t_mod = gamma * cls_t + beta

            # Kapı/gate analiziyle paralel — analiz/görselleştirme için sakla
            self.last_film_gamma = gamma.detach().mean().item()
            self.last_film_beta = beta.detach().mean().item()

            fused = cls_t_mod  # tek-akış çıktı, fusion_mlp'ye gerek yok
            if return_embedding:
                return fused
            return self.classifier(fused)
        else:
            # Concat ablation: cross-attention yok
            cls_t = h_t[:, 0, :]
            cls_f = h_f[:, 0, :]

        fused = self.fusion_mlp(torch.cat([cls_t, cls_f], dim=-1))  # (B, D)

        if return_embedding:
            return fused

        return self.classifier(fused)


class GLU(nn.Module):
    """ Gated Linear Unit - XGBoost'un dal ayrımı (tree split) yeteneğini DL'e taklit ettirir """

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        out = self.fc(x)
        val, gate = out.chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class DualStreamRiskModel(nn.Module):
    """
    XGBoost-killer: Fundamental veriyi time-series olarak değil, statik bir context
    olarak (MLP ile) değerlendirip, fiyat zaman serisini bu context ile koşullandıran
    (Conditioning) özel model.
    """

    def __init__(self,
                 tech_input_dim: int,
                 fund_input_dim: int,
                 seq_len: int = 20,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 dropout: float = 0.15):
        super().__init__()

        # 1. Fundamental Stream (Statik Profil Çıkarıcı)
        self.fund_encoder = nn.Sequential(
            nn.Linear(fund_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            GLU(hidden_dim)  # Gating yapısı ile gürültüyü filtrele
        )

        # 2. Technical Stream (Zaman Serisi Çıkarıcı)
        self.tech_proj = nn.Linear(tech_input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.tech_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Cross-Attention Fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 4. Sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            GLU(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_tech, x_fund, return_embedding=False):
        """
        Mevcut dataloader ile uyumlu olması için:
        x_tech: (B, T, tech_dim)
        x_fund: (B, T, fund_dim) -> MLP için (B, fund_dim)'e indirgenecek.
        """
        # DataLoader sequence yolladığı için sadece son günü (en güncel fundamental veri) alıyoruz.
        if x_fund.dim() == 3:
            fund_static = x_fund[:, -1, :]  # [Batch, fund_dim]
        else:
            fund_static = x_fund

        fund_emb = self.fund_encoder(fund_static)  # [Batch, hidden_dim]
        query = fund_emb.unsqueeze(1)  # [Batch, 1, hidden_dim]

        tech_emb = self.tech_proj(x_tech) + self.pos_encoder  # [Batch, Seq, hidden_dim]
        tech_encoded = self.tech_transformer(tech_emb)  # [Batch, Seq, hidden_dim]

        attn_out, _ = self.cross_attention(query=query, key=tech_encoded, value=tech_encoded)
        attn_out = attn_out.squeeze(1)  # [Batch, hidden_dim]

        tech_last = tech_encoded[:, -1, :]  # [Batch, hidden_dim]

        fused = torch.cat([attn_out, tech_last], dim=-1)  # [Batch, hidden_dim * 2]

        if return_embedding:
            return fused

        return self.classifier(fused)
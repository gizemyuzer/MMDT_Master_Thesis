import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss (logits üzerinde, numerik kararlı).

    Args:
        alpha: pozitif sınıf (Target=1) için ağırlık katsayısı.
               Sınıf dengesizliğinde n_neg/n_pos oranı önerilir.
        gamma: zor örnekleri (easy negatives'i bastırarak) öne çıkaran odak parametresi.
               gamma=0 -> Weighted BCE, gamma=2 -> orijinal Focal Loss paper.
        reduction: 'mean', 'sum' veya 'none'.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # P(true class)
        alpha_t = targets * self.alpha + (1 - targets) * 1.0
        focal = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


class DrawdownFocalLoss(nn.Module):
    """
    Kuyruk riskleri (tail risk) ve imbalanced sınıflar için özelleştirilmiş
    Asimetrik Focal Loss. False Negative'leri (kaçıran krizleri) ağır cezalandırır.
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        # alpha > 0.5 ise pozitif (kriz) sınıfına daha çok ağırlık verir.
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Tahmin olasılığı (doğru sınıfa ne kadar yakın)

        # Sınıf ağırlığı
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal formülü: (1-pt)^gamma * log(pt)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
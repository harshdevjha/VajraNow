import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatiotemporal sequence modeling."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        if state is None:
            hx = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
            cx = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
        else:
            hx, cx = state

        combined = torch.cat([x, hx], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * cx + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class FocalLoss(nn.Module):
    """Class-weighted focal loss for extreme thunderstorm/lightning class imbalance."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ResidualVajraGridUNet(nn.Module):
    """Residual U-Net with ConvLSTM bottleneck for lightning and severe convective nowcasting.

    Residual formulation:
    logit(P_lightning) = logit(P_advected_clipped) + f_lightning(X)
    logit(P_severe)    = logit(P_advected_severe_clipped) + f_severe(X)

    The final layer weights and biases are initialized to zero, ensuring that
    at step 0 or prior to training, the network output is EXACTLY identical to the
    calibrated physical advection baseline.
    """

    def __init__(
        self,
        in_channels: int = 18,  # 6 radar frames + 4 sat + 2 lightning + 4 env + 2 baseline
        hidden_dim: int = 32,
        out_lead_steps: int = 7,  # [10, 20, 30, 60, 90, 120, 180]
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_lead_steps = out_lead_steps

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # ConvLSTM Bottleneck
        self.convlstm = ConvLSTMCell(hidden_dim * 2, hidden_dim * 2, kernel_size=3)

        # Decoder
        self.up2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2 + in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Two forecasting heads: P(lightning) and P(severe convection) for all lead steps
        # Output channels: out_lead_steps for lightning, out_lead_steps for severe
        self.head_lightning = nn.Conv2d(hidden_dim, out_lead_steps, kernel_size=1)
        self.head_severe = nn.Conv2d(hidden_dim, out_lead_steps, kernel_size=1)

        # ZERO INITIALIZE final prediction heads for exact residual baseline match
        nn.init.zeros_(self.head_lightning.weight)
        nn.init.zeros_(self.head_lightning.bias)
        nn.init.zeros_(self.head_severe.weight)
        nn.init.zeros_(self.head_severe.bias)

    def forward(
        self,
        x: torch.Tensor,
        baseline_lightning_logit: torch.Tensor | None = None,
        baseline_severe_logit: torch.Tensor | None = None,
        radar_dropout_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)
            baseline_lightning_logit: Prior logit from advection baseline (B, Leads, H, W)
            baseline_severe_logit: Prior logit from advection baseline (B, Leads, H, W)
            radar_dropout_mask: Optional tensor to zero out radar channels during training/eval
        """
        if radar_dropout_mask is not None:
            # Apply radar channel dropout
            x = x * radar_dropout_mask

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck ConvLSTM
        h, _ = self.convlstm(p2)

        # Decoder
        d2 = self.up2(h)
        d2 = torch.cat([d2, p1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x], dim=1)
        feat = self.dec1(d1)

        # Residual perturbation logits
        delta_lightning = self.head_lightning(feat)
        delta_severe = self.head_severe(feat)

        if baseline_lightning_logit is not None:
            total_lightning_logit = baseline_lightning_logit + delta_lightning
        else:
            total_lightning_logit = delta_lightning

        if baseline_severe_logit is not None:
            total_severe_logit = baseline_severe_logit + delta_severe
        else:
            total_severe_logit = delta_severe

        prob_lightning = torch.sigmoid(total_lightning_logit)
        prob_severe = torch.sigmoid(total_severe_logit)

        return prob_lightning, prob_severe

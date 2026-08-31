"""Input/output standardization for the DML core — SP2.

Savine's point: differential ML dies without normalization, and gradient
labels must be rescaled by the chain rule — d(ŷ_scaled)/d(x_scaled) equals
``grad_scale`` (σ_x / σ_y, outer product) applied to dy/dx. Fitting uses
train-split statistics only so the val split stays untouched.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Standardizer:
    """Column-wise affine map for features (n_in) and labels (n_out)."""

    x_mean: torch.Tensor  # (n_in,)
    x_std: torch.Tensor   # (n_in,)
    y_mean: torch.Tensor  # (n_out,)
    y_std: torch.Tensor   # (n_out,)

    @classmethod
    def fit(cls, x: torch.Tensor, y: torch.Tensor) -> "Standardizer":
        """Fit on the train split; constant columns map to unit scale."""
        x_std = x.std(dim=0)
        y_std = y.std(dim=0)
        ones_x, ones_y = torch.ones_like(x_std), torch.ones_like(y_std)
        return cls(x.mean(dim=0), torch.where(x_std > 0, x_std, ones_x),
                   y.mean(dim=0), torch.where(y_std > 0, y_std, ones_y))

    def transform_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def inverse_x(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.x_std + self.x_mean

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std + self.y_mean

    @property
    def grad_scale(self) -> torch.Tensor:
        """(n_out, n_in) chain-rule factor for gradient labels: σ_x ⊗ (1/σ_y)."""
        return self.x_std[None, :] / self.y_std[:, None]

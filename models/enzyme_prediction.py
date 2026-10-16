from __future__ import annotations

import torch
from torch import nn


PH = 3
ESP = 4
MUTATION = 5

TASK_NAMES = ("kcat", "km", "kcat_km", "ph", "esp", "mutation")
TASK_TO_INDEX = {name: index for index, name in enumerate(TASK_NAMES)}

FITNESS_DIM = MUTATION


class EnzymePredictionHead(nn.Module):
    """Shared task trunk with independent adapters and Gaussian output heads."""

    def __init__(
        self, input_channels: int, hidden_channels: int, eps: float = 0.01, layout: str = "task_independent"
    ) -> None:
        super().__init__()
        self.layout = layout
        self.shared = nn.Sequential(nn.Linear(input_channels, hidden_channels), nn.SiLU())
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_channels, 64),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(64, hidden_channels),
            )
            for _ in TASK_NAMES
        )
        self.mu_heads = nn.ModuleList(nn.Linear(hidden_channels, 1) for _ in TASK_NAMES)
        self.sigma_heads = nn.ModuleList(nn.Linear(hidden_channels, 1) for _ in TASK_NAMES)
        self.eps = eps

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, fingerprint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all checkpoint-compatible task dimensions; select tasks at the caller."""
        shared = self.shared(fingerprint)
        task_features = [shared + adapter(shared) for adapter in self.adapters]
        mu = torch.cat([head(feature) for head, feature in zip(self.mu_heads, task_features)], dim=1)
        raw_sigma = torch.cat(
            [head(feature.detach()) for head, feature in zip(self.sigma_heads, task_features)], dim=1
        )
        sigma = (2.0 - self.eps) * torch.sigmoid(raw_sigma) + self.eps
        return mu, sigma


def task_distribution(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    task_index: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(task_index, int):
        task_index = torch.full((mu.size(0),), task_index, dtype=torch.long, device=mu.device)
    else:
        task_index = task_index.to(device=mu.device, dtype=torch.long).view(-1)
    rows = torch.arange(mu.size(0), device=mu.device)
    return mu[rows, task_index], sigma[rows, task_index]

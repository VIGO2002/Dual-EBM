"""Class-conditional energy decision module. Author: Chenwei He.

The module retains historical parameter names for checkpoint compatibility.
"""
import torch
from torch import nn


class DualEnergyHead(nn.Module):
    def __init__(self, in_features, hidden_dim=512, init_noise_std=0.02):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.1),
        )
        self.energy_real = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden_dim // 2, 1),
        )
        self.energy_fake = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden_dim // 2, 1),
        )
        self.noise_std = nn.Parameter(torch.tensor(float(init_noise_std)))

    def forward(self, x, training=True):
        h = self.shared_encoder(x)
        e_real, e_fake = self.energy_real(h), self.energy_fake(h)
        logits = torch.cat([-e_real, -e_fake], dim=1)
        if not training:
            return logits, e_real, e_fake
        sigma = torch.clamp(self.noise_std, min=0.001, max=0.05)
        perturbed = h + torch.randn_like(h) * sigma
        return logits, e_real, e_fake, self.energy_real(perturbed), self.energy_fake(perturbed)

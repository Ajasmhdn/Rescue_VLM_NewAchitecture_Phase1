import torch
import torch.nn as nn


class MultimodalProjector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=1024):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projector(x)
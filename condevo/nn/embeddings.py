import math
from torch import nn, linspace, sin, cos, cat, zeros_like, exp


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.register_buffer('freqs', exp(
            linspace(math.log(1e-4), math.log(1.0), steps=half)
        ), persistent=False)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))
    def forward(self, t):  # t: (B,) in [0, T-1]
        # scale to [0,1]
        t = t.float().unsqueeze(1)
        h = cat([sin(t * self.freqs), cos(t * self.freqs)], dim=1)
        if h.size(1) < self.dim:  # odd dims
            h = cat([h, zeros_like(h[:, :1])], dim=1)
        return self.proj(h)

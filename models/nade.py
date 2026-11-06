"""Neural Autoregressive Distribution Estimator (NADE).

Uses a shared feed-forward network and cumulative pre-activations so that all
autoregressive conditionals can be evaluated in a single forward pass.
"""
import math
import torch
import torch.nn as nn

class nade(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.randn(input_size, hidden_size) / math.sqrt(input_size))
        self.c = nn.Parameter(torch.zeros(hidden_size))
        self.V = nn.Parameter(torch.randn(hidden_size, input_size) / math.sqrt(hidden_size))
        self.b = nn.Parameter(torch.zeros(input_size))

    def _pre_activations(self, x:torch.Tensor) -> torch.Tensor:
        # [b, j , k] = [b, j , 1] * [1, j, k] 
        contrib = x.unsqueeze(-1) * self.W.unsqueeze(0)
        cum = contrib.cumsum(dim=1)
        zeros = torch.zeros_like(cum[:,:1,:])
        return torch.cat([zeros, cum[:,:-1,:]],dim=1)

    def _logits(self, x:torch.Tensor):
        l1 = self._pre_activations(x)
        # b d h
        l2 = torch.sigmoid(l1 + self.c)
        return (l2 * self.V.T.unsqueeze(0)).sum(-1) + self.b

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -nn.functional.binary_cross_entropy_with_logits(
            self._logits(x), x, reduction="none"
        )
    def forward(self, x):
        return self.log_prob(x)
    
    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        samples = torch.zeros(num_samples, self.input_size, device=device)
        a = torch.zeros(num_samples, self.hidden_size, device=device)
        for i in range(self.input_size):
            h = torch.sigmoid(a + self.c)
            logit = h @ self.V[:, i] + self.b[i]
            samples[:, i] = torch.bernoulli(torch.sigmoid(logit))
            if i < self.input_size - 1:
                a = a + torch.outer(samples[:, i], self.W[i])
        return samples

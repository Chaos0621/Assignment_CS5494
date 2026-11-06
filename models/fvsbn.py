"""Fully Visible Sigmoid Belief Network (FVSBN) on binarised images.

Each output pixel is a logistic regression over all previously ordered pixels;
autoregression is enforced with a strictly lower-triangular mask.
"""
import torch
import torch.nn as nn

input_size = 28*28

class Fvsbn(nn.Module  ):
    
    def __init__(self, input_size:int):
        super().__init__()
        self.input_size = input_size
        self.w = nn.Parameter(torch.zeros(input_size, input_size))
        self.b = nn.Parameter(torch.zeros(input_size))
        self.register_buffer("mask", torch.tril(torch.ones(input_size, input_size), diagonal=-1))

    def _logits(self, x:torch.Tensor):
        return x @ (self.w * self.mask).T + self.b

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._logits(x)
        return -nn.functional.binary_cross_entropy_with_logits(
            logits, x, reduction="none"
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.log_prob(x)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        samples = torch.zeros(num_samples, self.input_size, device=device)
        W_masked = self.w * self.mask
        for i in range(self.input_size):
            logit = samples @ W_masked[i] + self.b[i]
            samples[:, i] = torch.bernoulli(torch.sigmoid(logit))
        return samples

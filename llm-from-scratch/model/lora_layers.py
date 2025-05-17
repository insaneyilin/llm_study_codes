import math

import numpy as np
import torch
import torch.nn as nn


class LoRALayer(torch.nn.Module):

    def __init__(self, in_dim, out_dim, rank, alpha):
        """
        LoRA layer for low-rank adaptation.
        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            rank (int): Rank for low-rank adaptation.
            alpha (float): Scaling factor for LoRA.
        """
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        # Same initialization as nn.Linear for A.
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # Zero initialization for B, so A * B is zero at the start.
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):

    def __init__(self, linear, rank, alpha):
        """
        Linear layer with low-rank adaptation.
        Args:
            linear (torch.nn.Linear): Input linear layer.
            rank (int): Rank for low-rank adaptation.
            alpha (float): Scaling factor for LoRA.
        """
        super().__init__()
        self.linear = linear  # Original linear layer, used for forward pass, non-trainable.
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank,
                              alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the linear layer with a LoRALayer
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively replace in child modules
            replace_linear_with_lora(module, rank, alpha)

import math
from collections import defaultdict

import torch
from torch import nn


def gelu(x):
    """Gaussian Error Linear Unit activation function."""
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return cdf * x


class LayerNorm(nn.Module):
    """Layer Normalization module."""

    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.constant_(self.weight, 1.0)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        """Forward pass for layer normalization."""
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # Assign a unique ID to each module instance
    if not hasattr(module_instance, "_mygpt_instance_id"):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._mygpt_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return f"{module_name}.{module_instance._mygpt_instance_id}.{key}"


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value

import math

import torch


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Single-head attention.

    Args:
        q: [nq, d]
        k: [nk, d]
        v: [nk, d]
    Returns:
        output: [nq, d]
        probs: [nq, nk]
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    output = probs @ v
    return output, probs

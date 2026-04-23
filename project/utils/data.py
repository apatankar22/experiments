from dataclasses import dataclass

import torch


@dataclass
class KVCacheBatch:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


def generate_synthetic_kv(num_tokens: int, d_model: int, device: str = "cpu") -> KVCacheBatch:
    """Generate synthetic query/key/value tensors.

    Shapes:
        q: [1, d_model]
        k: [num_tokens, d_model]
        v: [num_tokens, d_model]
    """
    k = torch.randn(num_tokens, d_model, device=device)
    v = torch.randn(num_tokens, d_model, device=device)
    q = torch.randn(1, d_model, device=device)
    return KVCacheBatch(q=q, k=k, v=v)

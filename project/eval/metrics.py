import torch


def reconstruction_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.mean((x - y) ** 2).item()


def cosine_similarity_mean(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x_n = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    y_n = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (x_n * y_n).sum(dim=-1).mean().item()


def norm_preservation_error(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    nx = x.norm(dim=-1).clamp_min(eps)
    ny = y.norm(dim=-1)
    return torch.mean(torch.abs(nx - ny) / nx).item()


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.mean(torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)).item()

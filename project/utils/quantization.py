import torch


def uniform_quantize(x: torch.Tensor, bits: int, min_val: torch.Tensor | None = None, max_val: torch.Tensor | None = None):
    """Uniform affine quantization returning integer codes and metadata.

    Args:
        x: input tensor
        bits: quantization bit width
        min_val/max_val: optional bounds (broadcastable)
    """
    if bits <= 0:
        raise ValueError("bits must be positive")

    qmin = 0
    qmax = (1 << bits) - 1

    if min_val is None:
        min_val = x.amin(dim=-1, keepdim=True)
    if max_val is None:
        max_val = x.amax(dim=-1, keepdim=True)

    eps = torch.finfo(x.dtype).eps
    scale = (max_val - min_val).clamp_min(eps) / float(qmax - qmin)
    q = torch.round((x - min_val) / scale).clamp(qmin, qmax).to(torch.int32)
    return q, scale, min_val


def uniform_dequantize(q: torch.Tensor, scale: torch.Tensor, min_val: torch.Tensor):
    return q.to(scale.dtype) * scale + min_val


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

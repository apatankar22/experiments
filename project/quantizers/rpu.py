from dataclasses import dataclass

import torch

from project.utils.quantization import uniform_dequantize, uniform_quantize


@dataclass
class RPUEncoded:
    q: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    norms: torch.Tensor


class RPUQuantizer:
    """Random orthonormal preconditioning + uniform quantization."""

    def __init__(self, d_model: int, seed: int, normalize: bool = True):
        self.normalize = normalize
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        a = torch.randn(d_model, d_model, generator=g)
        q, _ = torch.linalg.qr(a, mode="reduced")
        self.Q = q

    def encode(self, x: torch.Tensor, bits: int) -> RPUEncoded:
        Q = self.Q.to(x.device, x.dtype)
        rot = x @ Q.T
        if self.normalize:
            norms = rot.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            rot_to_q = rot / norms
        else:
            norms = torch.ones_like(rot[..., :1])
            rot_to_q = rot

        q, scale, zero = uniform_quantize(rot_to_q, bits)
        return RPUEncoded(q=q, scale=scale, zero=zero, norms=norms)

    def decode(self, encoded: RPUEncoded, bits: int) -> torch.Tensor:
        del bits
        rot = uniform_dequantize(encoded.q, encoded.scale, encoded.zero)
        if self.normalize:
            rot = rot * encoded.norms
        Q = self.Q.to(rot.device, rot.dtype)
        return rot @ Q

    @staticmethod
    def bits_per_dim(bits: int) -> float:
        return float(bits)

from dataclasses import dataclass

import torch


@dataclass
class HS3Encoded:
    r: torch.Tensor
    phi_q: torch.Tensor
    theta_q: torch.Tensor
    orig_dim: int


class HS3Quantizer:
    """Hierarchical 3D spherical quantization using 3D blocks."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def _pad(self, x: torch.Tensor):
        d = x.shape[-1]
        pad = (3 - (d % 3)) % 3
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        return x, d

    def encode(self, x: torch.Tensor, bits: int) -> HS3Encoded:
        levels = 1 << bits
        x_pad, orig_dim = self._pad(x)
        blocks = x_pad.reshape(*x_pad.shape[:-1], -1, 3)
        bx, by, bz = blocks.unbind(dim=-1)

        r = torch.sqrt((blocks**2).sum(dim=-1) + self.eps)
        theta = torch.acos((bz / r).clamp(-1.0, 1.0))  # [0, pi]
        phi = torch.atan2(by, bx)  # [-pi, pi]
        phi = torch.where(phi < 0, phi + 2 * torch.pi, phi)  # [0, 2pi)

        phi_q = torch.round(phi / (2 * torch.pi) * (levels - 1)).clamp(0, levels - 1).to(torch.int16)
        theta_q = torch.round(theta / torch.pi * (levels - 1)).clamp(0, levels - 1).to(torch.int16)
        return HS3Encoded(r=r, phi_q=phi_q, theta_q=theta_q, orig_dim=orig_dim)

    def decode(self, encoded: HS3Encoded, bits: int) -> torch.Tensor:
        levels = 1 << bits
        phi = encoded.phi_q.to(encoded.r.dtype) / (levels - 1) * (2 * torch.pi)
        theta = encoded.theta_q.to(encoded.r.dtype) / (levels - 1) * torch.pi
        r = encoded.r

        sin_theta = torch.sin(theta)
        x = r * sin_theta * torch.cos(phi)
        y = r * sin_theta * torch.sin(phi)
        z = r * torch.cos(theta)

        out = torch.stack([x, y, z], dim=-1).reshape(*encoded.r.shape[:-1], -1)
        return out[..., : encoded.orig_dim]

    @staticmethod
    def bits_per_dim(bits: int) -> float:
        # radius kept in fp32 (32 bits) per 3D block + two quantized angles
        return (32 + 2 * bits) / 3.0

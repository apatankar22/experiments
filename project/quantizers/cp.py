from dataclasses import dataclass

import torch

from project.utils.quantization import uniform_dequantize, uniform_quantize


@dataclass
class CPEncoded:
    mag: torch.Tensor
    phase_q: torch.Tensor
    mag_q: torch.Tensor | None
    mag_scale: torch.Tensor | None
    mag_zero: torch.Tensor | None
    orig_dim: int


class CPQuantizer:
    """Complex phase-amplitude quantization on adjacent pairs."""

    def __init__(self, quantize_magnitude: bool = False):
        self.quantize_magnitude = quantize_magnitude

    def _pad(self, x: torch.Tensor):
        d = x.shape[-1]
        pad = d % 2
        if pad:
            x = torch.nn.functional.pad(x, (0, 1))
        return x, d

    def encode(self, x: torch.Tensor, bits: int) -> CPEncoded:
        levels = 1 << bits
        x_pad, orig_dim = self._pad(x)
        pairs = x_pad.reshape(*x_pad.shape[:-1], -1, 2)
        real = pairs[..., 0]
        imag = pairs[..., 1]

        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real)
        phase = torch.where(phase < 0, phase + 2 * torch.pi, phase)
        phase_q = torch.round(phase / (2 * torch.pi) * (levels - 1)).clamp(0, levels - 1).to(torch.int16)

        if self.quantize_magnitude:
            mag_q, mag_scale, mag_zero = uniform_quantize(mag, bits)
            mag_store = None
        else:
            mag_q, mag_scale, mag_zero = None, None, None
            mag_store = mag

        return CPEncoded(
            mag=mag_store,
            phase_q=phase_q,
            mag_q=mag_q,
            mag_scale=mag_scale,
            mag_zero=mag_zero,
            orig_dim=orig_dim,
        )

    def decode(self, encoded: CPEncoded, bits: int) -> torch.Tensor:
        levels = 1 << bits
        phase = encoded.phase_q.to(torch.float32) / (levels - 1) * (2 * torch.pi)

        if encoded.mag_q is None:
            mag = encoded.mag
        else:
            mag = uniform_dequantize(encoded.mag_q, encoded.mag_scale, encoded.mag_zero)

        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        out = torch.stack([real, imag], dim=-1).reshape(*phase.shape[:-1], -1)
        return out[..., : encoded.orig_dim]

    def bits_per_dim(self, bits: int) -> float:
        # per complex pair (2 dims): phase bits + optional mag bits or fp32 magnitude
        mag_bits = bits if self.quantize_magnitude else 32
        return (bits + mag_bits) / 2.0

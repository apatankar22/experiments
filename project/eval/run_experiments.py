from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from project.eval.metrics import (
    cosine_similarity_mean,
    kl_divergence,
    norm_preservation_error,
    reconstruction_mse,
)
from project.models.attention import scaled_dot_product_attention
from project.quantizers.cp import CPQuantizer
from project.quantizers.hs3 import HS3Quantizer
from project.quantizers.rpu import RPUQuantizer
from project.utils.data import generate_synthetic_kv
from project.utils.quantization import seed_everything

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


@dataclass
class EvalConfig:
    bits_list: list[int]
    seeds: list[int]
    dims: list[int]
    token_choices: list[int]
    output_dir: str = "project/results"


def _compress_pair(quantizer, k: torch.Tensor, v: torch.Tensor, bits: int):
    k_enc = quantizer.encode(k, bits)
    v_enc = quantizer.encode(v, bits)
    k_hat = quantizer.decode(k_enc, bits)
    v_hat = quantizer.decode(v_enc, bits)
    return k_hat, v_hat


def _jl_distortion(x: torch.Tensor, y: torch.Tensor, num_pairs: int = 64) -> float:
    n = x.shape[0]
    idx_a = torch.randint(0, n, (num_pairs,))
    idx_b = torch.randint(0, n, (num_pairs,))
    dx = (x[idx_a] - x[idx_b]).norm(dim=-1)
    dy = (y[idx_a] - y[idx_b]).norm(dim=-1)
    return (torch.abs(dy - dx) / dx.clamp_min(1e-8)).mean().item()


def run_experiments(cfg: EvalConfig):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    methods = [
        ("HS3", lambda d, s: HS3Quantizer(), lambda q, b: q.bits_per_dim(b)),
        ("CP-phase", lambda d, s: CPQuantizer(quantize_magnitude=False), lambda q, b: q.bits_per_dim(b)),
        ("CP-phase+mag", lambda d, s: CPQuantizer(quantize_magnitude=True), lambda q, b: q.bits_per_dim(b)),
        ("RPU-norm", lambda d, s: RPUQuantizer(d_model=d, seed=s + 13, normalize=True), lambda q, b: q.bits_per_dim(b)),
        ("RPU-no-norm", lambda d, s: RPUQuantizer(d_model=d, seed=s + 13, normalize=False), lambda q, b: q.bits_per_dim(b)),
    ]

    for seed in cfg.seeds:
        seed_everything(seed)
        for d_model in cfg.dims:
            for num_tokens in cfg.token_choices:
                batch = generate_synthetic_kv(num_tokens=num_tokens, d_model=d_model)
                ref_out, ref_probs = scaled_dot_product_attention(batch.q, batch.k, batch.v)

                for bits in cfg.bits_list:
                    for method_name, ctor, bpd_fn in methods:
                        quantizer = ctor(d_model, seed)
                        t0 = time.perf_counter()
                        k_hat, v_hat = _compress_pair(quantizer, batch.k, batch.v, bits)
                        out_hat, probs_hat = scaled_dot_product_attention(batch.q, k_hat, v_hat)
                        dt = time.perf_counter() - t0

                        rows.append(
                            {
                                "seed": seed,
                                "d_model": d_model,
                                "num_tokens": num_tokens,
                                "bits": bits,
                                "method": method_name,
                                "k_mse": reconstruction_mse(batch.k, k_hat),
                                "v_mse": reconstruction_mse(batch.v, v_hat),
                                "k_cos": cosine_similarity_mean(batch.k, k_hat),
                                "v_cos": cosine_similarity_mean(batch.v, v_hat),
                                "k_norm_err": norm_preservation_error(batch.k, k_hat),
                                "v_norm_err": norm_preservation_error(batch.v, v_hat),
                                "attn_out_mse": reconstruction_mse(ref_out, out_hat),
                                "attn_out_cos": cosine_similarity_mean(ref_out, out_hat),
                                "attn_kl": kl_divergence(ref_probs, probs_hat),
                                "jl_distortion": _jl_distortion(batch.k, k_hat),
                                "bits_per_dim": bpd_fn(quantizer, bits),
                                "compression_ratio": 32.0 / bpd_fn(quantizer, bits),
                                "runtime_ms": 1000.0 * dt,
                            }
                        )

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if plt is not None:
        _plot_metric(rows, out_dir, metric="k_mse", filename="mse_vs_bits.png", ylabel="Key reconstruction MSE")
        _plot_metric(rows, out_dir, metric="attn_out_mse", filename="attention_error_vs_bits.png", ylabel="Attention output MSE")
        _plot_metric(rows, out_dir, metric="attn_out_cos", filename="attention_cosine_vs_bits.png", ylabel="Attention output cosine")
    else:
        print("matplotlib not installed; skipping plot generation.")

    summary = summarize(rows)
    print(summary)
    return rows, summary


def _plot_metric(rows, out_dir: Path, metric: str, filename: str, ylabel: str):
    methods = sorted({r["method"] for r in rows})
    bits_list = sorted({r["bits"] for r in rows})

    plt.figure(figsize=(8, 5))
    for m in methods:
        ys = []
        for b in bits_list:
            vals = [r[metric] for r in rows if r["method"] == m and r["bits"] == b]
            ys.append(sum(vals) / len(vals))
        plt.plot(bits_list, ys, marker="o", label=m)

    plt.xlabel("Bit-width")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs bit-width")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=140)
    plt.close()


def summarize(rows: list[dict]) -> str:
    def best_for(bit_filter):
        subset = [r for r in rows if bit_filter(r["bits"]) ]
        grouped = {}
        for r in subset:
            grouped.setdefault(r["method"], []).append(r["attn_out_mse"])
        means = {k: sum(v) / len(v) for k, v in grouped.items()}
        best = min(means, key=means.get)
        return best, means[best]

    low_best, low_val = best_for(lambda b: b <= 4)
    high_best, high_val = best_for(lambda b: b >= 6)

    return (
        "\n=== Summary report ===\n"
        f"Best method (low-bit, <=4): {low_best} (mean attn MSE={low_val:.6f})\n"
        f"Best method (high-bit, >=6): {high_best} (mean attn MSE={high_val:.6f})\n"
        "RPU JL-style check: see jl_distortion column (lower is better).\n"
    )

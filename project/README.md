# Polar-Inspired KV-Cache Compression (PyTorch)

Implements and evaluates three KV-cache compression methods:

- **HS3**: Hierarchical 3D spherical quantization
- **CP**: Complex phase-amplitude quantization
- **RPU**: Random orthonormal preconditioning + uniform quantization

## Run

```bash
python -m pip install -r project/requirements.txt
python -m project.main
```

Outputs are written to `project/results/`:

- `results.csv`
- `mse_vs_bits.png`
- `attention_error_vs_bits.png`
- `attention_cosine_vs_bits.png`

## Notes

- Determinism is enforced via explicit seeding.
- Includes RPU ablation: with vs without normalization.
- Includes a JL-style empirical distortion metric via pairwise distances.

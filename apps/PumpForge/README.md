# PumpForge

PumpForge is a lightweight client layer that runs the 1D → JSON → 3D (inducer-only) pipeline using the core
modules under `src/`.

## CLI Usage

```bash
python -m apps.PumpForge.cli pump1d \
  --in apps/PumpForge/io/examples/pump1d_input.example.json \
  --out apps/PumpForge/io/examples/pump1d_output.example.json

python -m apps.PumpForge.cli pump3d \
  --in apps/PumpForge/io/examples/pump3d_input.example.json \
  --out apps/PumpForge/io/examples/pump3d_output_curves.example.json

python -m apps.PumpForge.cli all --case demo_case_001
```

The `all` command writes the pump1D output, embeds it into the pump3D input, and then runs the inducer-only pump3D
pipeline.

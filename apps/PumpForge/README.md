# PumpForge

PumpForge is a lightweight client layer that runs the 1D → JSON → 3D (inducer-only) pipeline using the core
modules under `src/`. Inputs are configured via JSON files, with a composed configuration approach that keeps
case-specific data separate from fixed constants.

## JSON include/merge support

You can compose inputs with `__include__` in any JSON file. Includes are resolved in order, deep-merged, and then
overridden by keys in the including file.

Example:

```json
{
  "__include__": [
    "apps/PumpForge/io/examples/pump1d_case.example.json",
    "apps/PumpForge/io/examples/pump1d_constants.example.json"
  ]
}
```

## CLI Usage

```bash
python -m apps.PumpForge.cli pump1d \
  --in apps/PumpForge/io/examples/pump1d_input.example.json \
  --out apps/PumpForge/io/examples/pump1d_output.example.json

python -m apps.PumpForge.cli pump3d \
  --in apps/PumpForge/io/examples/pump3d_input.example.json \
  --out apps/PumpForge/io/examples/pump3d_output_curves.example.json
```

You can also pass case/constant files directly:

```bash
python -m apps.PumpForge.cli pump1d \
  --case apps/PumpForge/io/examples/pump1d_case.example.json \
  --constants apps/PumpForge/io/examples/pump1d_constants.example.json

python -m apps.PumpForge.cli pump3d \
  --case apps/PumpForge/io/examples/inducer3d_case.example.json \
  --constants apps/PumpForge/io/examples/inducer3d_constants.example.json
```

The `all` command runs pump1D and then pump3D, embedding pump1D's `export_for_3d` in the pump3D input while keeping
pump3D inputs JSON-only:

```bash
python -m apps.PumpForge.cli all --case demo_case_001
```

## JSON-only inducer3D

Inducer3D is configured exclusively via `inducer3d_inputs` + `inducer3d_constants` in JSON. PumpForge adapts those
inputs into the minimal data structure needed by the inducer3D constructor without running inducer1D.

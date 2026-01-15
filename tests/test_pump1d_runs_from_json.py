import json
from math import isfinite
from pathlib import Path

from apps.PumpForge.io.json_codec import load_json
from apps.PumpForge.pipeline.run_pump1d import run_pump1d


def _assert_finite(value):
    if isinstance(value, (int, float)):
        assert isfinite(value)
    elif value is None:
        return
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)


def test_pump1d_runs_from_json(tmp_path):
    input_data = load_json(
        Path("apps/PumpForge/io/examples/pump1d_input.example.json")
    )
    output = run_pump1d(input_data)
    output_path = tmp_path / "pump1d_out.json"
    output_path.write_text(json.dumps(output), encoding="utf-8")

    assert output_path.exists()
    assert "performance" in output
    assert "geometry" in output
    assert "stations" in output

    _assert_finite(output)

from pathlib import Path

from apps.PumpForge.io.json_codec import load_json, save_json
from apps.PumpForge.pipeline.run_pump1d import run_pump1d
from apps.PumpForge.pipeline.run_pump3d import run_pump3d


def test_pumpforge_pipeline(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    examples = repo_root / "apps" / "PumpForge" / "io" / "examples"

    pump1d_input = examples / "pump1d_input.example.json"
    pump3d_input = examples / "pump3d_input.example.json"

    pump1d_output = tmp_path / "pump1d_output.json"
    pump1d_result = run_pump1d(pump1d_input, pump1d_output)

    assert pump1d_output.exists()
    pump1d_saved = load_json(pump1d_output)
    assert "export_for_3d" in pump1d_saved
    assert "export_for_3d" in pump1d_result

    pump3d_output_path = tmp_path / "pump3d_output.json"
    pump3d_result = run_pump3d(pump3d_input, pump3d_output_path)

    assert pump3d_output_path.exists()
    assert "curves" in pump3d_result
    assert "inducer" in pump3d_result["curves"]
    assert "hub_curve" in pump3d_result["curves"]["inducer"]
    assert pump3d_result["curves"]["impeller"]["mock"] is True
    assert pump3d_result["curves"]["volute"]["mock"] is True

    pump3d_payload = load_json(pump3d_input)
    pump3d_payload["from_pump1d"] = pump1d_result["export_for_3d"]
    pump3d_payload["inducer3d_inputs"]["geometry"].pop("tip_inlet_radius_m")
    pump3d_payload["inducer3d_inputs"]["geometry"].pop("tip_outlet_radius_m")

    pump3d_compat_input = tmp_path / "pump3d_input_compat.json"
    save_json(pump3d_compat_input, pump3d_payload)

    pump3d_compat_output = tmp_path / "pump3d_output_compat.json"
    pump3d_compat_result = run_pump3d(pump3d_compat_input, pump3d_compat_output)

    assert pump3d_compat_output.exists()
    assert "curves" in pump3d_compat_result
    assert "inducer" in pump3d_compat_result["curves"]

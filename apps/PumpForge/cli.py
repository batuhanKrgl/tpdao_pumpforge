from __future__ import annotations

import argparse
from pathlib import Path

from apps.PumpForge.io.json_codec import load_json, merge_dicts, save_json
from apps.PumpForge.pipeline.run_pump1d import run_pump1d
from apps.PumpForge.pipeline.run_pump3d import run_pump3d


def _default_paths() -> dict[str, Path]:
    base = Path(__file__).resolve().parent
    examples = base / "io" / "examples"
    return {
        "pump1d_in": examples / "pump1d_input.example.json",
        "pump1d_out": examples / "pump1d_output.example.json",
        "pump3d_in": examples / "pump3d_input.example.json",
        "pump3d_out": examples / "pump3d_output_curves.example.json",
    }


def _update_case_id(payload: dict, case_id: str | None) -> dict:
    if not case_id:
        return payload
    meta = payload.get("meta", {})
    meta["case_id"] = case_id
    payload["meta"] = meta
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="PumpForge CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pump1d_parser = subparsers.add_parser("pump1d", help="Run pump1D pipeline")
    pump1d_parser.add_argument("--in", dest="input_path")
    pump1d_parser.add_argument("--out", dest="output_path")

    pump3d_parser = subparsers.add_parser("pump3d", help="Run pump3D pipeline")
    pump3d_parser.add_argument("--in", dest="input_path")
    pump3d_parser.add_argument("--out", dest="output_path")

    all_parser = subparsers.add_parser("all", help="Run pump1D and pump3D pipeline")
    all_parser.add_argument("--case", dest="case_id")
    all_parser.add_argument("--pump1d-in", dest="pump1d_input_path")
    all_parser.add_argument("--pump1d-out", dest="pump1d_output_path")
    all_parser.add_argument("--pump3d-in", dest="pump3d_input_path")
    all_parser.add_argument("--pump3d-out", dest="pump3d_output_path")

    args = parser.parse_args()
    defaults = _default_paths()

    if args.command == "pump1d":
        input_path = Path(args.input_path or defaults["pump1d_in"])
        output_path = Path(args.output_path or defaults["pump1d_out"])
        run_pump1d(input_path, output_path)
        return

    if args.command == "pump3d":
        input_path = Path(args.input_path or defaults["pump3d_in"])
        output_path = Path(args.output_path or defaults["pump3d_out"])
        run_pump3d(input_path, output_path)
        return

    pump1d_input_path = Path(args.pump1d_input_path or defaults["pump1d_in"])
    pump1d_output_path = Path(args.pump1d_output_path or defaults["pump1d_out"])
    pump3d_input_path = Path(args.pump3d_input_path or defaults["pump3d_in"])
    pump3d_output_path = Path(args.pump3d_output_path or defaults["pump3d_out"])

    pump1d_output = run_pump1d(pump1d_input_path, pump1d_output_path)

    base_pump3d_input = load_json(pump3d_input_path)
    base_pump3d_input = _update_case_id(base_pump3d_input, args.case_id)
    pump3d_payload = merge_dicts(
        base_pump3d_input,
        {"from_pump1d": pump1d_output.get("export_for_3d", {})},
    )
    save_json(pump3d_input_path, pump3d_payload)

    run_pump3d(pump3d_input_path, pump3d_output_path)


if __name__ == "__main__":
    main()

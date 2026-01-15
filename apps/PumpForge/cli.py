import argparse
import json
from pathlib import Path

from apps.PumpForge.io.json_codec import load_json
from apps.PumpForge.pipeline.run_pump1d import run_pump1d


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PumpForge CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pump1d_parser = subparsers.add_parser("pump1d", help="Run pump1D solver")
    pump1d_parser.add_argument("--in", dest="input_path", required=True, help="Input JSON path")
    pump1d_parser.add_argument("--out", dest="output_path", required=True, help="Output JSON path")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "pump1d":
        input_data = load_json(Path(args.input_path))
        result = run_pump1d(input_data)
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()

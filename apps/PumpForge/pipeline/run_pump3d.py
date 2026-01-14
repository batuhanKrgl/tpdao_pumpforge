from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from apps.PumpForge.io.json_codec import load_json, save_json, to_jsonable
from apps.PumpForge.io.schemas import timestamp_utc
from src.Pump.pump1D import Inducer
from src.Pump.pump3D import Inducer3D


gravity = 9.805


def _validate_from_pump1d(data: Dict[str, Any]) -> None:
    required = ["omega_rad_s", "Q_m3_s", "rho_kg_m3", "shaft_radius_m", "inducer_head_req_m"]
    missing = [key for key in required if data.get(key) is None]
    if missing:
        raise ValueError(f"Missing from_pump1d keys: {', '.join(missing)}")


def _select_mid_curve(curves: List[np.ndarray]) -> np.ndarray:
    if not curves:
        return np.zeros((0, 3))
    return curves[len(curves) // 2]


def _curve_points(curve: np.ndarray) -> List[List[float]]:
    return to_jsonable(curve)


def _curve_group(curves: List[np.ndarray]) -> Dict[str, Any]:
    if not curves:
        return {"hub": [], "tip": [], "guides": []}
    hub_curve = curves[0]
    tip_curve = curves[-1]
    guide_curves = curves[1:-1]
    return {
        "hub": _curve_points(hub_curve),
        "tip": _curve_points(tip_curve),
        "guides": [_curve_points(curve) for curve in guide_curves],
    }


def _build_inducer_curves(inducer: Inducer3D) -> Dict[str, Any]:
    leading_curves = inducer.leading_edge_dict.get("curves", [])
    trailing_curves = inducer.trailing_edge_dict.get("curves", [])
    pressure_curves = inducer.foil_dict["pressure"]["curves"]
    suction_curves = inducer.foil_dict["suction"]["curves"]

    return {
        "hub_curve": _curve_points(inducer.hub),
        "tip_curve": _curve_points(inducer.tip),
        "leading_edge": _curve_points(_select_mid_curve(leading_curves)),
        "trailing_edge": _curve_points(_select_mid_curve(trailing_curves)),
        "blade_curves": {
            "pressure": _curve_group(pressure_curves),
            "suction": _curve_group(suction_curves),
        },
    }


def run_pump3d(input_path: str | Path, output_path: str | Path) -> Dict[str, Any]:
    payload = load_json(input_path)
    meta = payload.get("meta", {})
    from_pump1d = payload.get("from_pump1d", {})
    inducer_input = payload.get("inducer", {})
    impeller_input = payload.get("impeller", {})
    volute_input = payload.get("volute", {})

    _validate_from_pump1d(from_pump1d)

    inducer_results: Dict[str, Any] = {}
    if inducer_input.get("enabled", True):
        inducer = Inducer(
            vol_flow=from_pump1d["Q_m3_s"],
            head_req=from_pump1d["inducer_head_req_m"],
            omega=from_pump1d["omega_rad_s"],
            shaft_radius=from_pump1d["shaft_radius_m"],
            blade_number=inducer_input.get("n_blades", 3),
        )
        inducer3d = Inducer3D(inducer)
        inducer3d.initialize()
        inducer_results = _build_inducer_curves(inducer3d)

    output = {
        "meta": {
            "app": meta.get("app", "PumpForge"),
            "case_id": meta.get("case_id", ""),
            "stage": "pump3d",
            "component": "inducer",
            "units": meta.get("units", "SI"),
            "timestamp": timestamp_utc(),
        },
        "curves": {
            "inducer": inducer_results,
            "impeller": {
                "mock": True,
                **impeller_input.get("mock_outputs", {"status": "not_implemented_yet"}),
            },
            "volute": {
                "mock": True,
                **volute_input.get("mock_outputs", {"status": "not_implemented_yet"}),
            },
        },
    }

    save_json(output_path, to_jsonable(output))
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PumpForge pump3D pipeline")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    run_pump3d(args.input_path, args.output_path)

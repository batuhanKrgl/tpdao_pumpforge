from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from apps.PumpForge.io.json_codec import load_json, save_json, to_jsonable
from apps.PumpForge.io.schemas import INDUCER3D_DEFAULTS, timestamp_utc
from src.Pump import pump3D as pump3d_module
from src.Pump.pump3D import Inducer3D


def _validate_from_pump1d(data: Dict[str, Any]) -> None:
    required = ["omega_rad_s", "Q_m3_s", "rho_kg_m3", "shaft_radius_m", "inducer_head_req_m"]
    missing = [key for key in required if data.get(key) is None]
    if missing:
        raise ValueError(f"Missing from_pump1d keys: {', '.join(missing)}")


def _apply_from_pump1d(inducer_inputs: Dict[str, Any], from_pump1d: Dict[str, Any]) -> Dict[str, Any]:
    if not from_pump1d:
        return inducer_inputs
    inlet_flow = inducer_inputs.get("inlet_flow", {})
    inlet_flow.setdefault("Q_m3_s", from_pump1d.get("Q_m3_s"))
    inlet_flow.setdefault("rho_kg_m3", from_pump1d.get("rho_kg_m3"))
    inlet_flow.setdefault("omega_rad_s", from_pump1d.get("omega_rad_s"))
    inducer_inputs["inlet_flow"] = inlet_flow

    if "angles_deg" not in inducer_inputs and from_pump1d.get("angles_deg"):
        inducer_inputs["angles_deg"] = from_pump1d["angles_deg"]

    geometry = inducer_inputs.get("geometry", {})
    recommended = from_pump1d.get("recommended", {})
    geometry.setdefault("tip_inlet_radius_m", _half(recommended.get("D_in_m")))
    geometry.setdefault("tip_outlet_radius_m", _half(recommended.get("D_out_m")))
    inducer_inputs["geometry"] = geometry

    return inducer_inputs


def _half(value: float | None) -> float | None:
    if value is None:
        return None
    return 0.5 * value


@dataclass
class EdgeState:
    radius: float
    beta: float


@dataclass
class SpanState:
    inlet: EdgeState
    outlet: EdgeState


@dataclass
class Inducer1DLike:
    hub: SpanState
    tip: SpanState
    width: float
    l_over_t: float


def _build_inducer_like(inducer_inputs: Dict[str, Any], constants: Dict[str, Any]) -> Inducer1DLike:
    geometry = inducer_inputs.get("geometry", {})
    angles = inducer_inputs.get("angles_deg", {})
    blade = constants.get("blade", {})
    geometry_defaults = constants.get("geometry", {})

    hub_inlet_radius = geometry.get("hub_inlet_radius_m") or geometry.get("hub_radius_m")
    tip_inlet_radius = geometry.get("tip_inlet_radius_m") or geometry.get("tip_radius_m")
    hub_outlet_radius = geometry.get("hub_outlet_radius_m")
    tip_outlet_radius = geometry.get("tip_outlet_radius_m")
    width = geometry.get("width_m") or geometry_defaults.get("width_m")

    if hub_inlet_radius is None or tip_inlet_radius is None:
        raise ValueError("inducer3d_inputs.geometry hub/tip inlet radii are required")
    if hub_outlet_radius is None or tip_outlet_radius is None:
        raise ValueError("inducer3d_inputs.geometry hub/tip outlet radii are required")
    if width is None:
        raise ValueError("inducer3d_inputs.geometry.width_m is required")

    hub_inlet_beta = np.deg2rad(angles.get("hub_inlet_beta_deg", angles.get("beta1")))
    tip_inlet_beta = np.deg2rad(angles.get("tip_inlet_beta_deg", angles.get("beta1")))
    hub_outlet_beta = np.deg2rad(angles.get("hub_outlet_beta_deg", angles.get("beta2")))
    tip_outlet_beta = np.deg2rad(angles.get("tip_outlet_beta_deg", angles.get("beta2")))

    if None in [hub_inlet_beta, tip_inlet_beta, hub_outlet_beta, tip_outlet_beta]:
        raise ValueError("inducer3d_inputs.angles_deg requires beta1/beta2 or hub/tip betas")

    l_over_t = geometry.get(
        "l_over_t",
        geometry_defaults.get("l_over_t", INDUCER3D_DEFAULTS["geometry"]["l_over_t"]),
    )
    blade_thickness = blade.get("thickness_m", INDUCER3D_DEFAULTS["blade"]["thickness_m"])

    inducer_like = Inducer1DLike(
        hub=SpanState(inlet=EdgeState(hub_inlet_radius, hub_inlet_beta),
                      outlet=EdgeState(hub_outlet_radius, hub_outlet_beta)),
        tip=SpanState(inlet=EdgeState(tip_inlet_radius, tip_inlet_beta),
                      outlet=EdgeState(tip_outlet_radius, tip_outlet_beta)),
        width=width,
        l_over_t=l_over_t,
    )
    inducer_like.blade_thickness = blade_thickness
    return inducer_like


def _configure_inducer3d(inducer3d: Inducer3D, constants: Dict[str, Any], numerics: Dict[str, Any]) -> None:
    blade = constants.get("blade", {})
    numerics_defaults = constants.get("numerics", {})

    leading_ratio = blade.get(
        "leading_edge_ellipticity",
        INDUCER3D_DEFAULTS["blade"]["leading_edge_ellipticity"],
    )
    trailing_ratio = blade.get(
        "trailing_edge_ellipticity",
        INDUCER3D_DEFAULTS["blade"]["trailing_edge_ellipticity"],
    )
    inducer3d.leading_edge_dict["ratio"] = leading_ratio
    inducer3d.trailing_edge_dict["ratio"] = trailing_ratio

    thickness = blade.get("thickness_m", INDUCER3D_DEFAULTS["blade"]["thickness_m"])
    inducer3d.thickness_dict["hub"]["thickness"] = np.array([thickness, thickness])
    inducer3d.thickness_dict["tip"]["thickness"] = np.array([thickness, thickness])

    guides = numerics.get(
        "n_sections_spanwise",
        numerics_defaults.get("n_sections_spanwise", INDUCER3D_DEFAULTS["numerics"]["n_sections_spanwise"]),
    )
    if guides is not None:
        inducer3d.configure_guides(max(guides - 2, 0))


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
    inducer_inputs = payload.get("inducer3d_inputs", {})
    inducer_constants = payload.get("inducer3d_constants", {})
    numerics = payload.get("discretization", {})
    impeller_input = payload.get("impeller", {})
    volute_input = payload.get("volute", {})

    if from_pump1d:
        _validate_from_pump1d(from_pump1d)
        inducer_inputs = _apply_from_pump1d(inducer_inputs, from_pump1d)

    streamwise = numerics.get("n_points_streamwise")
    if streamwise is None:
        streamwise = inducer_constants.get("numerics", {}).get("n_points_streamwise")
    if streamwise:
        pump3d_module.nop = int(streamwise)

    inducer_results: Dict[str, Any] = {}
    if inducer_inputs.get("enabled", True):
        inducer_like = _build_inducer_like(inducer_inputs, inducer_constants)
        spanwise = numerics.get(
            "n_sections_spanwise",
            inducer_constants.get("numerics", {}).get(
                "n_sections_spanwise",
                INDUCER3D_DEFAULTS["numerics"]["n_sections_spanwise"],
            ),
        )
        inducer3d = Inducer3D(inducer_like, number_of_guides=max(spanwise - 2, 0))
        _configure_inducer3d(inducer3d, inducer_constants, numerics)
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
        "constants_used": {
            "inducer3d_constants": inducer_constants,
            "discretization": numerics,
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

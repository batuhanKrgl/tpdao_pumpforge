from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from apps.PumpForge.io.json_codec import load_json, save_json, to_jsonable
from apps.PumpForge.io.schemas import PUMP1D_DEFAULTS, PUMP1D_OUTPUT_KEYS, timestamp_utc
from src.Pump.pump1D import Pump
from src.fluid import Fluid


def _build_pump_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    operating_point = payload.get("operating_point", {})
    fluid_info = payload.get("fluid", {})
    inlet = payload.get("inlet", {})
    geometry = payload.get("geometry", {})
    constants = payload.get("pump1d_constants", {})

    rho = fluid_info.get("rho_kg_m3")
    if rho is None:
        raise ValueError("fluid.rho_kg_m3 is required")

    q_m3_s = operating_point.get("Q_m3_s")
    h_m = operating_point.get("H_m")
    if q_m3_s is None or h_m is None:
        raise ValueError("operating_point.Q_m3_s and operating_point.H_m are required")

    inlet_pressure = inlet.get("p_in_Pa")
    if inlet_pressure is None:
        raise ValueError("inlet.p_in_Pa is required")

    inlet_diameter = geometry.get("inlet_diameter_m")
    outlet_diameter = geometry.get("outlet_diameter_m")
    if inlet_diameter is None or outlet_diameter is None:
        raise ValueError("geometry.inlet_diameter_m and geometry.outlet_diameter_m are required")

    gravity = constants.get("gravity_m_s2", PUMP1D_DEFAULTS["gravity_m_s2"])
    pressure_required = inlet_pressure + rho * gravity * h_m
    mass_flow = q_m3_s * rho

    configuration_defaults = constants.get("configuration", PUMP1D_DEFAULTS["configuration"])
    inlet_defaults = constants.get("inlet", PUMP1D_DEFAULTS["inlet"])

    inlet_alpha = inlet.get("alpha_deg", inlet_defaults.get("alpha_deg"))
    if inlet_alpha is None:
        raise ValueError("inlet.alpha_deg or pump1d_constants.inlet.alpha_deg is required")

    return {
        "mass_flow": mass_flow,
        "pressure_required": pressure_required,
        "inlet_pressure": inlet_pressure,
        "alpha": inlet_alpha,
        "double_suction": geometry.get("double_suction", configuration_defaults.get("double_suction")),
        "second_stage": geometry.get("second_stage", configuration_defaults.get("second_stage")),
        "inlet_radius": inlet_diameter / 2,
        "outlet_radius": outlet_diameter / 2,
        "shaft_radius": geometry.get("shaft_radius_m", constants.get("shaft_radius_m")),
    }


def _build_fluid(payload: Dict[str, Any]) -> Fluid:
    fluid_info = payload.get("fluid", {})
    fluid = Fluid()
    fluid.update_properties("name", fluid_info.get("name", ""))
    fluid.update_properties("density", fluid_info.get("rho_kg_m3", 0))
    fluid.update_properties("dynamicViscosity", fluid_info.get("mu_Pa_s", 0))
    fluid.update_properties("vaporPressure", fluid_info.get("p_vapor_Pa", 0))
    return fluid


def _average_deg(a_rad: float, b_rad: float) -> float:
    return float(np.rad2deg(0.5 * (a_rad + b_rad)))


def _build_output(payload: Dict[str, Any], pump: Pump) -> Dict[str, Any]:
    meta = payload.get("meta", {})
    operating_point = payload.get("operating_point", {})
    fluid_info = payload.get("fluid", {})
    constants = payload.get("pump1d_constants", {})

    impeller = pump.impeller
    inlet_beta_hub = float(np.rad2deg(impeller.hub.inlet.beta))
    inlet_beta_tip = float(np.rad2deg(impeller.tip.inlet.beta))
    outlet_beta_hub = float(np.rad2deg(impeller.hub.outlet.beta))
    outlet_beta_tip = float(np.rad2deg(impeller.tip.outlet.beta))
    torque = None
    if pump.omega:
        torque = pump.shaft_power / pump.omega if pump.shaft_power is not None else None

    export_for_3d = {
        "omega_rad_s": pump.omega,
        "Q_m3_s": pump.vol_flow,
        "H_m": pump.head_rise,
        "rho_kg_m3": pump.fluid.density,
        "shaft_radius_m": pump.shaft_radius,
        "inducer_head_req_m": pump.npsh_required or max(pump.head_rise * 0.1, 0.1),
        "recommended": {
            "D_in_m": 2 * impeller.tip.inlet.radius,
            "D_out_m": 2 * impeller.tip.outlet.radius,
        },
        "angles_deg": {
            "beta1": _average_deg(impeller.hub.inlet.beta, impeller.tip.inlet.beta),
            "beta2": _average_deg(impeller.hub.outlet.beta, impeller.tip.outlet.beta),
        },
    }

    output = {
        "meta": {
            "app": meta.get("app", "PumpForge"),
            "case_id": meta.get("case_id", ""),
            "stage": "pump1d",
            "units": meta.get("units", "SI"),
            "timestamp": timestamp_utc(),
        },
        "operating_point": {
            "Q_m3_s": operating_point.get("Q_m3_s"),
            "H_m": operating_point.get("H_m"),
            "n_rpm": operating_point.get("n_rpm"),
        },
        "performance": {
            "eta_hydraulic": pump.hydraulic_efficiency,
            "eta_total": pump.total_efficiency,
            "power_W": pump.shaft_power,
            "torque_Nm": torque,
            "npsh_required_m": pump.npsh_required,
        },
        "sizing": {
            "D_in_m": 2 * impeller.tip.inlet.radius,
            "D_out_m": 2 * impeller.tip.outlet.radius,
            "b_in_m": impeller.width,
            "b_out_m": impeller.outlet_width,
            "hub_radius_in_m": impeller.hub.inlet.radius,
            "hub_radius_out_m": impeller.hub.outlet.radius,
            "tip_radius_in_m": impeller.tip.inlet.radius,
            "tip_radius_out_m": impeller.tip.outlet.radius,
        },
        "velocity_triangles": {
            "inlet": {"beta_hub_deg": inlet_beta_hub, "beta_tip_deg": inlet_beta_tip},
            "outlet": {"beta_hub_deg": outlet_beta_hub, "beta_tip_deg": outlet_beta_tip},
        },
        "export_for_3d": export_for_3d,
        "constants_used": {
            "gravity_m_s2": constants.get("gravity_m_s2", PUMP1D_DEFAULTS["gravity_m_s2"]),
            "defaults": constants,
        },
        "validation": to_jsonable(PUMP1D_OUTPUT_KEYS),
        "fluid": {
            "name": fluid_info.get("name", ""),
            "rho_kg_m3": pump.fluid.density,
            "mu_Pa_s": pump.fluid.dynamicViscosity,
            "p_vapor_Pa": pump.fluid.vaporPressure,
        },
    }
    return output


def run_pump1d(input_path: str | Path, output_path: str | Path) -> Dict[str, Any]:
    payload = load_json(input_path)
    operating_point = payload.get("operating_point", {})
    rpm = operating_point.get("n_rpm")
    if rpm is None:
        raise ValueError("operating_point.n_rpm is required")

    pump_dict = _build_pump_dict(payload)
    fluid = _build_fluid(payload)
    pump = Pump(rpm=rpm, pump_dict=pump_dict, fluid=fluid)
    output = _build_output(payload, pump)
    save_json(output_path, to_jsonable(output))
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PumpForge pump1D pipeline")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    run_pump1d(args.input_path, args.output_path)

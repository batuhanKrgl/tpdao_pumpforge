from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

from src.pump1d.constants import default_constants


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class FluidProps:
    density_kg_m3: float
    kinematic_viscosity_m2_s: float
    vapor_pressure_Pa: float

    @classmethod
    def from_dict(cls, data: dict) -> "FluidProps":
        return cls(
            density_kg_m3=float(data["density_kg_m3"]),
            kinematic_viscosity_m2_s=float(data["kinematic_viscosity_m2_s"]),
            vapor_pressure_Pa=float(data["vapor_pressure_Pa"]),
        )


@dataclass(frozen=True)
class SecondaryFlows:
    station_flows: dict[str, list[float]]

    @classmethod
    def from_dict(cls, data: dict | None) -> "SecondaryFlows | None":
        if data is None:
            return None
        return cls(station_flows={key: list(value) for key, value in data.items()})


@dataclass(frozen=True)
class CaseInput:
    rpm: float
    omega_rad_s: float
    mass_flow_kg_s: float
    pressure_required_Pa: float
    inlet_pressure_Pa: float
    inlet_radius_m: float
    outlet_radius_m: float
    alpha_inlet_rad: float
    shaft_radius_m: float | None
    double_suction: bool
    second_stage: bool
    fluid: FluidProps
    secondary_flows: SecondaryFlows | None

    @classmethod
    def from_dict(cls, data: dict) -> "CaseInput":
        rpm = data.get("rpm")
        omega = data.get("omega_rad_s")
        if rpm is None and omega is None:
            raise ValueError("Either rpm or omega_rad_s must be provided.")
        if rpm is None:
            rpm = float(omega) * 30.0 / pi
        rpm = float(rpm)
        omega = float(omega) if omega is not None else rpm * pi / 30.0

        alpha_rad = data.get("alpha_inlet_rad")
        alpha_deg = data.get("alpha_inlet_deg")
        if alpha_rad is None and alpha_deg is None:
            raise ValueError("alpha_inlet_rad or alpha_inlet_deg is required.")
        if alpha_rad is None:
            alpha_rad = float(alpha_deg) * pi / 180.0
        alpha_rad = float(alpha_rad)

        return cls(
            rpm=rpm,
            omega_rad_s=omega,
            mass_flow_kg_s=float(data["mass_flow_kg_s"]),
            pressure_required_Pa=float(data["pressure_required_Pa"]),
            inlet_pressure_Pa=float(data["inlet_pressure_Pa"]),
            inlet_radius_m=float(data["inlet_radius_m"]),
            outlet_radius_m=float(data["outlet_radius_m"]),
            alpha_inlet_rad=alpha_rad,
            shaft_radius_m=float(data["shaft_radius_m"]) if data.get("shaft_radius_m") is not None else None,
            double_suction=bool(data["double_suction"]),
            second_stage=bool(data["second_stage"]),
            fluid=FluidProps.from_dict(data["fluid"]),
            secondary_flows=SecondaryFlows.from_dict(data.get("secondary_flows")),
        )


@dataclass(frozen=True)
class Pump1DInput:
    case: CaseInput
    constants: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict) -> "Pump1DInput":
        constants = _deep_merge(default_constants(), data.get("constants", {}))
        return cls(
            case=CaseInput.from_dict(data["case"]),
            constants=constants,
        )


@dataclass(frozen=True)
class Pump1DResult:
    performance: dict[str, Any]
    geometry: dict[str, Any]
    stations: dict[str, Any]
    assumptions_used: dict[str, Any]
    schema_version: str = "1.0"

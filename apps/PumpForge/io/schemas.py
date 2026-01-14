from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(frozen=True)
class Pump1DOutputSchema:
    meta: Dict[str, Any]
    operating_point: Dict[str, Any]
    performance: Dict[str, Any]
    sizing: Dict[str, Any]
    velocity_triangles: Dict[str, Any]
    export_for_3d: Dict[str, Any]
    constants_used: Dict[str, Any]


PUMP1D_OUTPUT_KEYS = Pump1DOutputSchema(
    meta={"app": "", "case_id": "", "stage": "pump1d", "units": "SI", "timestamp": ""},
    operating_point={"Q_m3_s": None, "H_m": None, "n_rpm": None},
    performance={"eta_hydraulic": None, "eta_total": None, "power_W": None, "torque_Nm": None},
    sizing={
        "D_in_m": None,
        "D_out_m": None,
        "b_in_m": None,
        "b_out_m": None,
        "hub_radius_in_m": None,
        "hub_radius_out_m": None,
        "tip_radius_in_m": None,
        "tip_radius_out_m": None,
    },
    velocity_triangles={
        "inlet": {"beta_hub_deg": None, "beta_tip_deg": None},
        "outlet": {"beta_hub_deg": None, "beta_tip_deg": None},
    },
    export_for_3d={
        "omega_rad_s": None,
        "Q_m3_s": None,
        "H_m": None,
        "rho_kg_m3": None,
        "shaft_radius_m": None,
        "inducer_head_req_m": None,
        "recommended": {"D_in_m": None, "D_out_m": None},
        "angles_deg": {"beta1": None, "beta2": None},
    },
    constants_used={"gravity_m_s2": None, "defaults": {}},
)


PUMP1D_DEFAULTS = {
    "gravity_m_s2": 9.805,
    "configuration": {"double_suction": False, "second_stage": False},
    "inlet": {"alpha_deg": 90.0},
}


INDUCER3D_DEFAULTS = {
    "blade": {
        "n_blades": 3,
        "thickness_m": 0.0008,
        "leading_edge_ellipticity": 1.5,
        "trailing_edge_ellipticity": 2.0,
    },
    "geometry": {"l_over_t": 2.0},
    "numerics": {"n_sections_spanwise": 7},
}


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

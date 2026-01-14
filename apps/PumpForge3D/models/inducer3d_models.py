from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Inducer3DInputs:
    shaft_radius_m: float = 0.01
    hub_inlet_radius_m: float = 0.015
    tip_inlet_radius_m: float = 0.03
    hub_outlet_radius_m: float = 0.012
    tip_outlet_radius_m: float = 0.028
    blade_count: int = 3
    axial_length_m: float = 0.05
    beta_inlet_rad: float = 0.0
    beta_outlet_rad: float = 0.0
    thickness_profile: Dict[str, Any] = field(default_factory=dict)
    n_spanwise: int = 11
    n_points: int = 51
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Curve3D:
    name: str
    points: List[List[float]]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Inducer3DOutputs:
    hub_curve: Curve3D
    tip_curve: Curve3D
    leading_edge_curve: Optional[Curve3D] = None
    trailing_edge_curve: Optional[Curve3D] = None
    blade_section_curves: Optional[List[Curve3D]] = None
    notes: Optional[str] = None


@dataclass
class Inducer3DState:
    inputs: Inducer3DInputs
    outputs: Optional[Inducer3DOutputs] = None
    version: str = "0.1"

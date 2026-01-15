from __future__ import annotations

from dataclasses import dataclass
from math import pi, tan

from src.pump1d.rotor_base import RotorBase
from src.pump1d.triangles import TriangleState, compute_blockage, compute_slip_factor


@dataclass
class TriangleSummary:
    radius_m: float
    alpha_rad: float | None
    beta_blade_rad: float | None
    c_m: float
    c_u: float
    u: float


@dataclass
class ImpellerResult:
    hub_inlet: TriangleSummary
    hub_outlet: TriangleSummary
    tip_inlet: TriangleSummary
    tip_outlet: TriangleSummary
    head_rise_m: float
    outlet_width_m: float
    width_m: float
    blade_number: int
    thickness_matrix: list[list[float]]


class Impeller(RotorBase):
    def __init__(
        self,
        vol_flow_m3_s: float,
        head_req_m: float,
        omega_rad_s: float,
        shaft_radius_m: float,
        inlet_radius_m: float,
        outlet_radius_m: float,
        alpha_inlet_rad: float,
        blade_number: int,
        thickness_matrix: list[list[float]],
        beta_blade_outlet_rad: float,
        blockage_coeff: float,
        design_incidence_coeff: float,
        gravity_m_s2: float,
    ) -> None:
        super().__init__()
        self.vol_flow_m3_s = vol_flow_m3_s
        self.head_req_m = head_req_m
        self.omega_rad_s = omega_rad_s
        self.shaft_radius_m = shaft_radius_m
        self.inlet_radius_m = inlet_radius_m
        self.outlet_radius_m = outlet_radius_m
        self.alpha_inlet_rad = alpha_inlet_rad
        self.blade_number = blade_number
        self.thickness_matrix = thickness_matrix
        self.beta_blade_outlet_rad = beta_blade_outlet_rad
        self.blockage_coeff = blockage_coeff
        self.design_incidence_coeff = design_incidence_coeff
        self.gravity_m_s2 = gravity_m_s2
        self.result: ImpellerResult | None = None

    def calc_inlet(self) -> TriangleSummary:
        inlet_area = pi * (self.inlet_radius_m**2 - self.shaft_radius_m**2)
        inlet_area = max(inlet_area, 1e-8)
        c_m = self.vol_flow_m3_s / inlet_area
        u = self.inlet_radius_m * self.omega_rad_s
        tan_alpha = tan(self.alpha_inlet_rad)
        c_u = c_m / (1e-12 if abs(tan_alpha) < 1e-12 else tan_alpha)
        return TriangleSummary(
            radius_m=self.inlet_radius_m,
            alpha_rad=self.alpha_inlet_rad,
            beta_blade_rad=None,
            c_m=c_m,
            c_u=c_u,
            u=u,
        )

    def calc_outlet(self) -> TriangleSummary:
        outlet_area = pi * (self.outlet_radius_m**2 - self.shaft_radius_m**2)
        outlet_area = max(outlet_area, 1e-8)
        c_m = self.vol_flow_m3_s / outlet_area
        thickness_m = float(self.thickness_matrix[1][0])
        blockage = compute_blockage(self.blade_number, thickness_m, self.outlet_radius_m, self.beta_blade_outlet_rad)
        slip = compute_slip_factor(
            self.beta_blade_outlet_rad,
            self.blade_number,
            self.inlet_radius_m,
            self.outlet_radius_m,
            True,
        )
        u = self.outlet_radius_m * self.omega_rad_s
        tan_beta = tan(self.beta_blade_outlet_rad)
        c_u = u * (slip - c_m * blockage / u / (1e-12 if abs(tan_beta) < 1e-12 else tan_beta))
        return TriangleSummary(
            radius_m=self.outlet_radius_m,
            alpha_rad=None,
            beta_blade_rad=self.beta_blade_outlet_rad,
            c_m=c_m,
            c_u=c_u,
            u=u,
        )

    def solve(self) -> ImpellerResult:
        hub_inlet = self.calc_inlet()
        tip_inlet = self.calc_inlet()
        hub_outlet = self.calc_outlet()
        tip_outlet = self.calc_outlet()

        head_rise = (hub_outlet.c_u * hub_outlet.u - hub_inlet.c_u * hub_inlet.u) / self.gravity_m_s2
        outlet_width = self.vol_flow_m3_s / (2 * pi * self.outlet_radius_m * hub_outlet.c_m)
        width_m = outlet_width

        self.state.solved = True
        self.result = ImpellerResult(
            hub_inlet=hub_inlet,
            hub_outlet=hub_outlet,
            tip_inlet=tip_inlet,
            tip_outlet=tip_outlet,
            head_rise_m=head_rise,
            outlet_width_m=outlet_width,
            width_m=width_m,
            blade_number=self.blade_number,
            thickness_matrix=self.thickness_matrix,
        )
        return self.result


def triangle_to_state(summary: TriangleSummary) -> TriangleState:
    return TriangleState(
        radius_m=summary.radius_m,
        omega_rad_s=0.0,
        blade_number=0,
        thickness_m=0.0,
        incidence_rad=0.0,
        alpha_rad=summary.alpha_rad,
        beta_blade_rad=summary.beta_blade_rad,
        inlet_radius_m=summary.radius_m,
        c_m=summary.c_m,
        c_u=summary.c_u,
        u=summary.u,
    )

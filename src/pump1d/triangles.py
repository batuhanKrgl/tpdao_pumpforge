from __future__ import annotations

from dataclasses import dataclass
from math import exp, sin, sqrt, tan


@dataclass
class TriangleState:
    radius_m: float
    omega_rad_s: float
    blade_number: int
    thickness_m: float
    incidence_rad: float
    alpha_rad: float | None
    beta_blade_rad: float | None
    inlet_radius_m: float
    c_m: float
    c_u: float | None = None
    u: float | None = None
    blockage: float | None = None
    slip_factor: float | None = None


def compute_blockage(blade_number: int, thickness_m: float, radius_m: float, beta_blade_rad: float) -> float:
    return (1 - blade_number * thickness_m / 3.141592653589793 / radius_m / 2 / sin(beta_blade_rad)) ** -1


def compute_slip_factor(
    beta_blade_rad: float,
    blade_number: int,
    inlet_radius_m: float,
    radius_m: float,
    is_outlet: bool,
) -> float:
    if not is_outlet:
        return 0.0
    epsilon_lim = exp(-8.16 * sin(beta_blade_rad) / blade_number)
    d1m_star = inlet_radius_m / radius_m
    if d1m_star < epsilon_lim:
        k_w = 1.0
    else:
        k_w = 1 - ((d1m_star - epsilon_lim) / (1 - epsilon_lim)) ** 3
    f1 = 0.98
    return f1 * (1 - sqrt(sin(beta_blade_rad)) / blade_number ** 0.7) * k_w


def compute_triangle_inlet(state: TriangleState) -> TriangleState:
    state.u = state.radius_m * state.omega_rad_s
    if state.alpha_rad is None:
        raise ValueError("alpha_rad is required for inlet triangle")
    tan_alpha = tan(state.alpha_rad)
    state.c_u = state.c_m / (1e-12 if abs(tan_alpha) < 1e-12 else tan_alpha)
    return state


def compute_triangle_outlet(state: TriangleState) -> TriangleState:
    state.u = state.radius_m * state.omega_rad_s
    if state.beta_blade_rad is None:
        raise ValueError("beta_blade_rad is required for outlet triangle")
    state.blockage = compute_blockage(state.blade_number, state.thickness_m, state.radius_m, state.beta_blade_rad)
    state.slip_factor = compute_slip_factor(
        state.beta_blade_rad,
        state.blade_number,
        state.inlet_radius_m,
        state.radius_m,
        True,
    )
    tan_beta = tan(state.beta_blade_rad)
    state.c_u = state.u * (state.slip_factor - state.c_m * state.blockage / state.u / (1e-12 if abs(tan_beta) < 1e-12 else tan_beta))
    return state

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np

from apps.PumpForge3D.meridional.bezier import bezier_curve, sample_bezier


@dataclass
class MeridionalBezierParams:
    Lz_m: float
    hub_r_in_m: float
    hub_r_out_m: float
    b_in_m: float
    b_out_m: float
    hub_theta_in_deg: float
    hub_theta_out_deg: float
    tip_theta_in_deg: float
    tip_theta_out_deg: float
    hub_handle_in: float
    hub_handle_out: float
    tip_handle_in: float
    tip_handle_out: float
    hub_p2_z_norm: float
    hub_p2_r_norm: float
    tip_p2_z_norm: float
    tip_p2_r_norm: float
    le_ctrl_z_offset_m: float
    te_ctrl_z_offset_m: float
    le_hub_t_norm: float
    le_tip_t_norm: float
    te_hub_t_norm: float
    te_tip_t_norm: float
    n_curve_points: int


@dataclass
class MeridionalSection2D:
    hub_curve: np.ndarray
    tip_curve: np.ndarray
    leading_edge: np.ndarray
    trailing_edge: np.ndarray
    hub_ctrl: np.ndarray
    tip_ctrl: np.ndarray
    le_ctrl: np.ndarray
    te_ctrl: np.ndarray


def _clamp_handle(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_norm(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate_params(params: MeridionalBezierParams) -> None:
    if params.Lz_m <= 0:
        raise ValueError("Meridional length Lz must be positive.")
    if params.hub_r_in_m <= 0 or params.hub_r_out_m <= 0:
        raise ValueError("Hub radii must be positive.")
    if params.b_in_m <= 0 or params.b_out_m <= 0:
        raise ValueError("Channel heights must be positive.")
    tip_r_in = params.hub_r_in_m + params.b_in_m
    tip_r_out = params.hub_r_out_m + params.b_out_m
    if tip_r_in <= params.hub_r_in_m or tip_r_out <= params.hub_r_out_m:
        raise ValueError("Tip radii must be larger than hub radii at inlet/outlet.")
    if params.n_curve_points < 50:
        raise ValueError("n_curve_points must be at least 50.")


def _unit_vector(theta_deg: float) -> np.ndarray:
    theta_rad = radians(theta_deg)
    return np.array([cos(theta_rad), sin(theta_rad)], dtype=float)


def build_section(params: MeridionalBezierParams) -> MeridionalSection2D:
    _validate_params(params)

    Lz = params.Lz_m
    hub_handle_in = _clamp_handle(params.hub_handle_in)
    hub_handle_out = _clamp_handle(params.hub_handle_out)
    tip_handle_in = _clamp_handle(params.tip_handle_in)
    tip_handle_out = _clamp_handle(params.tip_handle_out)
    n_points = max(50, int(params.n_curve_points))

    hub_p0 = np.array([0.0, params.hub_r_in_m], dtype=float)
    hub_p4 = np.array([Lz, params.hub_r_out_m], dtype=float)
    hub_p1 = hub_p0 + hub_handle_in * Lz * _unit_vector(params.hub_theta_in_deg)
    hub_p3 = hub_p4 - hub_handle_out * Lz * _unit_vector(params.hub_theta_out_deg)
    hub_z_norm = _clamp_norm(params.hub_p2_z_norm)
    hub_r_norm = _clamp_norm(params.hub_p2_r_norm)
    hub_p2_z = hub_z_norm * Lz
    hub_r_min = min(params.hub_r_in_m, params.hub_r_out_m)
    hub_r_max = max(params.hub_r_in_m, params.hub_r_out_m)
    hub_p2_r = hub_r_min + hub_r_norm * (hub_r_max - hub_r_min)
    hub_p2 = np.array([hub_p2_z, hub_p2_r], dtype=float)
    hub_ctrl = np.vstack([hub_p0, hub_p1, hub_p2, hub_p3, hub_p4])

    tip_r_in = params.hub_r_in_m + params.b_in_m
    tip_r_out = params.hub_r_out_m + params.b_out_m
    tip_p0 = np.array([0.0, tip_r_in], dtype=float)
    tip_p4 = np.array([Lz, tip_r_out], dtype=float)
    tip_p1 = tip_p0 + tip_handle_in * Lz * _unit_vector(params.tip_theta_in_deg)
    tip_p3 = tip_p4 - tip_handle_out * Lz * _unit_vector(params.tip_theta_out_deg)
    tip_z_norm = _clamp_norm(params.tip_p2_z_norm)
    tip_r_norm = _clamp_norm(params.tip_p2_r_norm)
    tip_p2_z = tip_z_norm * Lz
    tip_r_min = min(tip_r_in, tip_r_out)
    tip_r_max = max(tip_r_in, tip_r_out)
    tip_p2_r = tip_r_min + tip_r_norm * (tip_r_max - tip_r_min)
    tip_p2 = np.array([tip_p2_z, tip_p2_r], dtype=float)
    tip_ctrl = np.vstack([tip_p0, tip_p1, tip_p2, tip_p3, tip_p4])

    le_hub_t = _clamp_norm(params.le_hub_t_norm)
    le_tip_t = _clamp_norm(params.le_tip_t_norm)
    le_hub_point = bezier_curve(hub_ctrl, np.array([le_hub_t]))[0]
    le_tip_point = bezier_curve(tip_ctrl, np.array([le_tip_t]))[0]
    le_r_mid = 0.5 * (le_hub_point[1] + le_tip_point[1])
    le_z_mid = 0.5 * (le_hub_point[0] + le_tip_point[0])
    le_ctrl_point = np.array([le_z_mid + params.le_ctrl_z_offset_m, le_r_mid], dtype=float)
    le_ctrl = np.vstack([le_hub_point, le_ctrl_point, le_tip_point])

    te_hub_t = _clamp_norm(params.te_hub_t_norm)
    te_tip_t = _clamp_norm(params.te_tip_t_norm)
    te_hub_point = bezier_curve(hub_ctrl, np.array([te_hub_t]))[0]
    te_tip_point = bezier_curve(tip_ctrl, np.array([te_tip_t]))[0]
    te_r_mid = 0.5 * (te_hub_point[1] + te_tip_point[1])
    te_z_mid = 0.5 * (te_hub_point[0] + te_tip_point[0])
    te_ctrl_point = np.array([te_z_mid + params.te_ctrl_z_offset_m, te_r_mid], dtype=float)
    te_ctrl = np.vstack([te_hub_point, te_ctrl_point, te_tip_point])

    hub_curve = sample_bezier(hub_ctrl, n_points=n_points)
    tip_curve = sample_bezier(tip_ctrl, n_points=n_points)
    leading_edge = sample_bezier(le_ctrl, n_points=n_points)
    trailing_edge = sample_bezier(te_ctrl, n_points=n_points)

    return MeridionalSection2D(
        hub_curve=hub_curve,
        tip_curve=tip_curve,
        leading_edge=leading_edge,
        trailing_edge=trailing_edge,
        hub_ctrl=hub_ctrl,
        tip_ctrl=tip_ctrl,
        le_ctrl=le_ctrl,
        te_ctrl=te_ctrl,
    )

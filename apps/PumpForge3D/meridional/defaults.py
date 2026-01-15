from __future__ import annotations

from apps.PumpForge3D.meridional.section_model import MeridionalBezierParams


def default_params() -> MeridionalBezierParams:
    return MeridionalBezierParams(
        Lz_m=0.06,
        hub_r_in_m=0.015,
        hub_r_out_m=0.020,
        b_in_m=0.012,
        b_out_m=0.008,
        hub_theta_in_deg=90.0,
        hub_theta_out_deg=30.0,
        tip_theta_in_deg=85.0,
        tip_theta_out_deg=25.0,
        hub_handle_in=0.2,
        hub_handle_out=0.2,
        tip_handle_in=0.2,
        tip_handle_out=0.2,
        hub_p2_z_norm=0.5,
        hub_p2_r_norm=0.5,
        tip_p2_z_norm=0.5,
        tip_p2_r_norm=0.5,
        le_ctrl_z_offset_m=-0.002,
        te_ctrl_z_offset_m=0.002,
        le_hub_t_norm=0.0,
        le_tip_t_norm=0.0,
        te_hub_t_norm=1.0,
        te_tip_t_norm=1.0,
        n_curve_points=200,
    )

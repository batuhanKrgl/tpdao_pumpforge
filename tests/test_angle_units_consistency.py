from math import isfinite, pi
from pathlib import Path

from apps.PumpForge.io.json_codec import load_json
from src.pump1d.models import Pump1DInput
from src.pump1d.triangles import compute_blockage, compute_slip_factor


def test_angle_units_are_radians():
    input_data = load_json(Path("apps/PumpForge/io/examples/pump1d_input.example.json"))
    pump_input = Pump1DInput.from_dict(input_data)
    alpha = pump_input.case.alpha_inlet_rad
    assert 0 < alpha < pi

    slip = compute_slip_factor(
        beta_blade_rad=alpha,
        blade_number=6,
        inlet_radius_m=0.02,
        radius_m=0.03,
        is_outlet=True,
    )
    blockage = compute_blockage(6, 0.002, 0.03, alpha)
    assert isfinite(slip)
    assert isfinite(blockage)

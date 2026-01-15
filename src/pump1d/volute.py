from dataclasses import dataclass

from src.pump1d.rotor_base import RotorBase


@dataclass
class VoluteResult:
    inlet_radius_m: float
    inlet_height_m: float
    outlet_radius_m: float
    cutwater_radius_m: float
    diffuser_angle_rad: float
    loss_coefficient: float


class Volute(RotorBase):
    def __init__(
        self,
        inlet_radius_m: float,
        outlet_radius_m: float,
        inlet_height_m: float,
        cutwater_radius_m: float,
        diffuser_angle_rad: float,
        loss_coefficient: float,
    ) -> None:
        super().__init__()
        self.inlet_radius_m = inlet_radius_m
        self.outlet_radius_m = outlet_radius_m
        self.inlet_height_m = inlet_height_m
        self.cutwater_radius_m = cutwater_radius_m
        self.diffuser_angle_rad = diffuser_angle_rad
        self.loss_coefficient = loss_coefficient
        self.result: VoluteResult | None = None

    def solve(self) -> VoluteResult:
        self.state.solved = True
        self.result = VoluteResult(
            inlet_radius_m=self.inlet_radius_m,
            inlet_height_m=self.inlet_height_m,
            outlet_radius_m=self.outlet_radius_m,
            cutwater_radius_m=self.cutwater_radius_m,
            diffuser_angle_rad=self.diffuser_angle_rad,
            loss_coefficient=self.loss_coefficient,
        )
        return self.result

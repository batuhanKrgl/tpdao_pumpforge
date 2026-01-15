from __future__ import annotations

from math import pi, sqrt

from src.pump1d.impeller import Impeller
from src.pump1d.inducer import Inducer
from src.pump1d.models import Pump1DInput, Pump1DResult
from src.pump1d.stations import Station
from src.pump1d.volute import Volute


def _default_secondary_flows(mass_flow_kg_s: float, ratio: float) -> dict[str, list[float]]:
    return {
        "station_3": [-ratio * mass_flow_kg_s],
        "station_2": [-ratio * mass_flow_kg_s, -ratio * mass_flow_kg_s],
        "station_2c": [ratio * mass_flow_kg_s],
        "station_2b": [ratio * mass_flow_kg_s],
        "station_2a": [ratio * mass_flow_kg_s],
        "station_1": [ratio * mass_flow_kg_s],
        "station_1s": [ratio * mass_flow_kg_s],
        "station_1i": [ratio * mass_flow_kg_s],
        "station_1is": [-ratio * mass_flow_kg_s],
    }


def _calc_hydraulic_efficiency(nss: float) -> float:
    nss_brennen = nss * 15850.32**0.5 / 3.28083**0.75 / 2734.6
    if nss_brennen < 0.8:
        return (
            0.41989
            + 2.1524 * nss_brennen
            - 3.1434 * nss_brennen**2
            + 1.5679 * nss_brennen**3
        )
    return 1.020 - 0.120 * nss_brennen


def _calc_shaft_radius(h_w: float, rpm: float, s_a: float) -> float:
    shaft_list = [value * 1e-3 / 2 for value in [10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55]]
    omega = rpm * pi / 30
    shaft_radius = 0.5 * 1.5 * (16 * h_w / omega / pi / s_a) ** (1 / 3)
    for candidate in shaft_list:
        if candidate - shaft_radius > 0:
            return candidate
    return shaft_list[-1]


class PumpSolver:
    def __init__(self, pump_input: Pump1DInput) -> None:
        self.pump_input = pump_input

    def solve(self) -> Pump1DResult:
        case = self.pump_input.case
        constants = self.pump_input.constants
        gravity = constants["gravity"]
        mech_eff = constants["mechanical_efficiency"]

        mass_flow = case.mass_flow_kg_s
        density = case.fluid.density_kg_m3
        vol_flow = mass_flow / density

        pressure_rise = case.pressure_required_Pa - case.inlet_pressure_Pa
        head_rise = pressure_rise / density / gravity

        specific_speed = case.rpm * sqrt(vol_flow) / (head_rise ** 0.75) if head_rise > 0 else 0.0
        hydraulic_eff = _calc_hydraulic_efficiency(specific_speed) if head_rise > 0 else 0.0

        secondary_flows = (
            case.secondary_flows.station_flows
            if case.secondary_flows
            else _default_secondary_flows(mass_flow, constants["default_secondary_flow_ratio"])
        )

        volumetric_eff = mass_flow / (mass_flow + abs(sum(secondary_flows["station_3"])))
        total_eff = hydraulic_eff * volumetric_eff * mech_eff

        hydraulic_work = vol_flow * pressure_rise
        shaft_radius = case.shaft_radius_m
        if shaft_radius is None:
            shaft_radius = _calc_shaft_radius(
                h_w=hydraulic_work,
                s_a=constants.get("stress_allowable_Pa", 200e6),
                rpm=case.rpm,
            )

        inlet_area = pi * (case.inlet_radius_m ** 2 - shaft_radius ** 2)
        outlet_area = pi * (case.outlet_radius_m ** 2 - shaft_radius ** 2)
        inlet_area = max(inlet_area, 1e-8)
        outlet_area = max(outlet_area, 1e-8)

        blade_number = int(constants["geometry"]["blade_number"])
        beta_blade_outlet_rad = constants["geometry"]["impeller_beta_blade_outlet_deg"] * pi / 180.0
        impeller = Impeller(
            vol_flow_m3_s=vol_flow,
            head_req_m=head_rise,
            omega_rad_s=case.omega_rad_s,
            shaft_radius_m=shaft_radius,
            inlet_radius_m=case.inlet_radius_m,
            outlet_radius_m=case.outlet_radius_m,
            alpha_inlet_rad=case.alpha_inlet_rad,
            blade_number=blade_number,
            thickness_matrix=constants["geometry"]["thickness_matrix_default"],
            beta_blade_outlet_rad=beta_blade_outlet_rad,
            blockage_coeff=constants["correlations"]["blockage_coeff"],
            design_incidence_coeff=constants["correlations"]["design_incidence_coeff"],
            gravity_m_s2=gravity,
        )
        impeller_result = impeller.solve()

        inducer = Inducer(
            vol_flow_m3_s=vol_flow,
            head_req_m=head_rise,
            omega_rad_s=case.omega_rad_s,
            shaft_radius_m=shaft_radius,
            inlet_radius_m=case.inlet_radius_m,
            outlet_radius_m=case.inlet_radius_m,
            alpha_inlet_rad=case.alpha_inlet_rad,
            blade_number=blade_number,
            thickness_matrix=constants["geometry"]["thickness_matrix_default"],
            beta_blade_inlet_rad=constants["geometry"]["inducer_beta_blade_inlet_deg"] * pi / 180.0,
            beta_blade_outlet_rad=constants["geometry"]["inducer_beta_blade_outlet_deg"] * pi / 180.0,
            l_over_t=constants["geometry"]["inducer_l_over_t"],
            gravity_m_s2=gravity,
        )
        inducer_result = inducer.solve()

        volute = Volute(
            inlet_radius_m=case.outlet_radius_m,
            outlet_radius_m=case.outlet_radius_m * 1.2,
            inlet_height_m=impeller_result.outlet_width_m,
            cutwater_radius_m=case.outlet_radius_m * 1.05,
            diffuser_angle_rad=constants["geometry"]["volute_diffuser_angle_deg"] * pi / 180.0,
            loss_coefficient=0.02,
        )
        volute_result = volute.solve()

        npsh_available = (case.inlet_pressure_Pa - case.fluid.vapor_pressure_Pa) / density / gravity
        npsh_required = 0.1 * head_rise

        shaft_power = hydraulic_work / (hydraulic_eff * volumetric_eff * mech_eff) if total_eff > 0 else 0.0

        station_0 = Station(
            name="station_0",
            total_pressure_Pa=case.inlet_pressure_Pa,
            static_pressure_Pa=case.inlet_pressure_Pa - 0.5 * density * (vol_flow / inlet_area) ** 2,
            mass_flow_kg_s=mass_flow,
        )
        station_3 = Station(
            name="station_3",
            total_pressure_Pa=case.pressure_required_Pa,
            static_pressure_Pa=case.pressure_required_Pa - constants["station_pressure_drop_Pa"],
            mass_flow_kg_s=mass_flow,
        )
        station_2 = Station(
            name="station_2",
            total_pressure_Pa=case.pressure_required_Pa - constants["station_pressure_drop_Pa"],
            static_pressure_Pa=case.pressure_required_Pa - 2 * constants["station_pressure_drop_Pa"],
            mass_flow_kg_s=mass_flow,
        )
        station_1 = Station(
            name="station_1",
            total_pressure_Pa=case.inlet_pressure_Pa + pressure_rise * 0.5,
            static_pressure_Pa=case.inlet_pressure_Pa + pressure_rise * 0.5,
            mass_flow_kg_s=mass_flow,
        )

        performance = {
            "head_rise_m": head_rise,
            "npsh_available_m": npsh_available,
            "npsh_required_m": npsh_required,
            "hydraulic_efficiency": hydraulic_eff,
            "total_efficiency": total_eff,
            "shaft_power_W": shaft_power,
            "volumetric_efficiency": volumetric_eff,
            "pressure_out_Pa": station_3.total_pressure_Pa,
        }

        geometry = {
            "shaft_radius_m": shaft_radius,
            "inducer": {
                "hub_inlet_radius_m": inducer_result.hub_inlet.radius_m,
                "tip_inlet_radius_m": inducer_result.tip_inlet.radius_m,
                "hub_outlet_radius_m": inducer_result.hub_outlet.radius_m,
                "tip_outlet_radius_m": inducer_result.tip_outlet.radius_m,
                "width_m": constants["geometry"]["inducer_axial_length_m"],
                "blade_number": inducer_result.blade_number,
                "l_over_t": inducer_result.l_over_t,
                "triangles": {
                    "hub_inlet": {
                        "alpha_rad": inducer_result.hub_inlet.alpha_rad,
                        "beta_blade_rad": inducer_result.hub_inlet.beta_blade_rad,
                        "c_m": inducer_result.hub_inlet.c_m,
                        "c_u": inducer_result.hub_inlet.c_u,
                        "u": inducer_result.hub_inlet.u,
                    },
                    "hub_outlet": {
                        "alpha_rad": None,
                        "beta_blade_rad": inducer_result.hub_outlet.beta_blade_rad,
                        "c_m": inducer_result.hub_outlet.c_m,
                        "c_u": inducer_result.hub_outlet.c_u,
                        "u": inducer_result.hub_outlet.u,
                    },
                },
                "thickness_matrix": inducer_result.thickness_matrix,
                "thickness_array": inducer_result.thickness_matrix[1],
                "axial_length_m": constants["geometry"]["inducer_axial_length_m"],
            },
            "impeller": {
                "hub_inlet_radius_m": impeller_result.hub_inlet.radius_m,
                "tip_inlet_radius_m": impeller_result.tip_inlet.radius_m,
                "hub_outlet_radius_m": impeller_result.hub_outlet.radius_m,
                "tip_outlet_radius_m": impeller_result.tip_outlet.radius_m,
                "outlet_width_m": impeller_result.outlet_width_m,
                "width_m": impeller_result.width_m,
                "blade_number": impeller_result.blade_number,
                "beta_blade_outlet_rad": impeller_result.hub_outlet.beta_blade_rad,
                "triangles": {
                    "hub_inlet": {
                        "alpha_rad": impeller_result.hub_inlet.alpha_rad,
                        "beta_blade_rad": impeller_result.hub_inlet.beta_blade_rad,
                        "c_m": impeller_result.hub_inlet.c_m,
                        "c_u": impeller_result.hub_inlet.c_u,
                        "u": impeller_result.hub_inlet.u,
                    },
                    "hub_outlet": {
                        "alpha_rad": None,
                        "beta_blade_rad": impeller_result.hub_outlet.beta_blade_rad,
                        "c_m": impeller_result.hub_outlet.c_m,
                        "c_u": impeller_result.hub_outlet.c_u,
                        "u": impeller_result.hub_outlet.u,
                    },
                },
                "thickness_matrix": impeller_result.thickness_matrix,
                "thickness_array": impeller_result.thickness_matrix[1],
                "npsh_required_m": npsh_required,
            },
            "volute": {
                "inlet_radius_m": volute_result.inlet_radius_m,
                "inlet_height_m": volute_result.inlet_height_m,
                "outlet_radius_m": volute_result.outlet_radius_m,
                "cutwater_radius_m": volute_result.cutwater_radius_m,
                "diffuser_angle_rad": volute_result.diffuser_angle_rad,
                "loss_coefficient": volute_result.loss_coefficient,
            },
        }

        stations = {
            station_0.name: station_0.as_dict(),
            station_1.name: station_1.as_dict(),
            station_2.name: station_2.as_dict(),
            station_3.name: station_3.as_dict(),
        }

        assumptions_used = {
            "constants": constants,
            "secondary_flows": secondary_flows,
        }

        return Pump1DResult(
            performance=performance,
            geometry=geometry,
            stations=stations,
            assumptions_used=assumptions_used,
        )

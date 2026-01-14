from copy import deepcopy

DEFAULT_CONSTANTS = {
    "gravity": 9.80665,
    "mechanical_efficiency": 0.90,
    "station_pressure_drop_Pa": 100000.0,
    "default_secondary_flow_ratio": 0.025,
    "stress_allowable_Pa": 200e6,
    "solver": {
        "max_iter_impeller_opt": 100,
        "max_iter_match_inducer": 100,
        "tol_head_m": 10.0,
        "step_up": 1.01,
        "step_down": 0.99,
    },
    "correlations": {
        "blockage_coeff": 1.6,
        "design_incidence_coeff": 0.3,
    },
    "geometry": {
        "blade_number": 6,
        "inducer_l_over_t": 2.5,
        "impeller_beta_blade_outlet_deg": 23.0,
        "inducer_beta_blade_inlet_deg": 20.0,
        "inducer_beta_blade_outlet_deg": 25.0,
        "thickness_matrix_default": [[0.0, 1.0], [0.002, 0.002]],
        "inducer_axial_length_m": 0.02,
        "impeller_axial_length_m": 0.02,
        "volute_diffuser_angle_deg": 10.0,
    },
}


def default_constants() -> dict:
    return deepcopy(DEFAULT_CONSTANTS)

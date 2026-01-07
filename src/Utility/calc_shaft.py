import numpy as np


def min_impeller_r(shaft_r):
    shaft_diameter_list = np.array([10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55]) * 1e-3
    t2 = np.array([1.4, 1.8, 2.3, 2.3, 2.8, 3.3, 3.3, 3.3, 3.3, 3.8, 3.8, 4.3]) * 1e-3
    b = np.array([3, 4, 5, 5, 6, 8, 8, 10, 12, 14, 14, 16]) * 1e-3

    arg = np.where(shaft_diameter_list == shaft_r * 2)
    return np.round(np.sqrt((shaft_r + t2[arg]) ** 2 + (b[arg] / 2) ** 2) + 2e-3, 4)[0]


def calc_shaft_radius(h_w, rpm, s_a):
    shaftList = np.array([10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55]) * 1e-3 / 2
    omega = rpm * np.pi / 30
    # Gulich sf348 7.1.2
    shaft_radius = 0.5 * 1.5 * (16 * h_w / omega / np.pi / s_a) ** 1 / 3
    return shaftList[np.where(shaftList - shaft_radius > 0)[0][0]]


def calc_hydraulic_efficiency(nss):
    nssimport numpy as np


def min_impeller_r(shaft_r):
    shaft_diameter_list = np.array([10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55]) * 1e-3
    t2 = np.array([1.4, 1.8, 2.3, 2.3, 2.8, 3.3, 3.3, 3.3, 3.3, 3.8, 3.8, 4.3]) * 1e-3
    b = np.array([3, 4, 5, 5, 6, 8, 8, 10, 12, 14, 14, 16]) * 1e-3

    arg = np.where(shaft_diameter_list == shaft_r * 2)
    return np.round(np.sqrt((shaft_r + t2[arg]) ** 2 + (b[arg] / 2) ** 2) + 2e-3, 4)[0]


def calc_shaft_radius(h_w, rpm, s_a):
    shaftList = np.array([10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55]) * 1e-3 / 2
    omega = rpm * np.pi / 30
    # Gulich sf348 7.1.2
    shaft_radius = 0.5 * 1.5 * (16 * h_w / omega / np.pi / s_a) ** 1 / 3
    return shaftList[np.where(shaftList - shaft_radius > 0)[0][0]]


def calc_hydraulic_efficiency(nss):
    nss_brennen = nss * 15850.32**0.5 / 3.28083**0.75 / 2734.6
    if nss_brennen < 0.8:
        h_e = 0.41989 +\
              2.1524 * nss_brennen -\
              3.1434 * nss_brennen ** 2 +\
              1.5679 * nss_brennen ** 3
    else:
        h_e = 1.020 - 0.120 * nss_brennen

    return h_e


def calc_shaft_radius_from_dict(tp_dict, rpm):
    s_a = 200e6
    m_e = 0.95
    v_e = 0.95
    ox_dict = tp_dict["oxidizer_pump"]
    fu_dict = tp_dict["fuel_pump"]

    v_f_ox = ox_dict["mass_flow"] / ox_dict["fluid"]["density"]
    v_f_fu = fu_dict["mass_flow"] / fu_dict["fluid"]["density"]
    nss_ox = rpm * np.sqrt(v_f_ox) / np.power((ox_dict["pressure_required"] - ox_dict["inlet_pressure"]) / ox_dict["fluid"]["density"] / 9.805, 0.75)
    nss_fu = rpm * np.sqrt(v_f_fu) / np.power((fu_dict["pressure_required"] - fu_dict["inlet_pressure"]) / fu_dict["fluid"]["density"] / 9.805, 0.75)

    h_e_ox = calc_hydraulic_efficiency(nss_ox)
    h_e_fu = calc_hydraulic_efficiency(nss_fu)

    h_w_o = (ox_dict["pressure_required"] - ox_dict["inlet_pressure"]) * v_f_ox / h_e_ox / m_e/ v_e
    h_w_f = (fu_dict["pressure_required"] - fu_dict["inlet_pressure"]) * v_f_fu / h_e_fu / m_e/ v_e

    shaftList = np.array([10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55]) * 1e-3 / 2
    omega = rpm * np.pi / 30
    # Gulich sf348 7.1.2

    shaft_radius_fu = 0.5 * 1.5 * (16 * (h_w_f + h_w_o) / omega / np.pi / s_a) ** (1/3)
    shaft_radius_ox = 0.5 * 1.5 * (16 * (h_w_o) / omega / np.pi / s_a) ** (1 / 3)

    arg_ox = np.where(shaftList - shaft_radius_ox > 0)[0][0]
    arg_fu = np.where(shaftList - shaft_radius_fu > 0)[0][0]

    if shaft_radius_ox < shaftList[arg_ox-1] + (shaftList[arg_ox] - shaftList[arg_ox-1])/3:
        tp_dict["oxidizer_pump"]["shaft_radius"]= shaftList[arg_ox-1]
    else:
        tp_dict["oxidizer_pump"]["shaft_radius"] = shaftList[arg_ox]

    if shaft_radius_fu < shaftList[arg_fu-1] + (shaftList[arg_fu] - shaftList[arg_fu-1])/3:
        tp_dict["fuel_pump"]["shaft_radius"]= shaftList[arg_fu-1]
    else:
        tp_dict["fuel_pump"]["shaft_radius"] = shaftList[arg_fu]

    return tp_dict_brennen = nss * 15850.32**0.5 / 3.28083**0.75 / 2734.6
    if nss_brennen < 0.8:
        h_e = 0.41989 +\
              2.1524 * nss_brennen -\
              3.1434 * nss_brennen ** 2 +\
              1.5679 * nss_brennen ** 3
    else:
        h_e = 1.020 - 0.120 * nss_brennen

    return h_e


def calc_shaft_radius_from_dict(tp_dict, rpm):
    s_a = 200e6
    m_e = 0.95
    v_e = 0.95
    ox_dict = tp_dict["oxidizer_pump"]
    fu_dict = tp_dict["fuel_pump"]

    v_f_ox = ox_dict["mass_flow"] / ox_dict["fluid"]["density"]
    v_f_fu = fu_dict["mass_flow"] / fu_dict["fluid"]["density"]
    nss_ox = rpm * np.sqrt(v_f_ox) / np.power((ox_dict["pressure_required"] - ox_dict["inlet_pressure"]) / ox_dict["fluid"]["density"] / 9.805, 0.75)
    nss_fu = rpm * np.sqrt(v_f_fu) / np.power((fu_dict["pressure_required"] - fu_dict["inlet_pressure"]) / fu_dict["fluid"]["density"] / 9.805, 0.75)

    h_e_ox = calc_hydraulic_efficiency(nss_ox)
    h_e_fu = calc_hydraulic_efficiency(nss_fu)

    h_w_o = (ox_dict["pressure_required"] - ox_dict["inlet_pressure"]) * v_f_ox / h_e_ox / m_e/ v_e
    h_w_f = (fu_dict["pressure_required"] - fu_dict["inlet_pressure"]) * v_f_fu / h_e_fu / m_e/ v_e

    shaftList = np.array([10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55]) * 1e-3 / 2
    omega = rpm * np.pi / 30
    # Gulich sf348 7.1.2

    shaft_radius_fu = 0.5 * 1.5 * (16 * (h_w_f + h_w_o) / omega / np.pi / s_a) ** (1/3)
    shaft_radius_ox = 0.5 * 1.5 * (16 * (h_w_o) / omega / np.pi / s_a) ** (1 / 3)

    arg_ox = np.where(shaftList - shaft_radius_ox > 0)[0][0]
    arg_fu = np.where(shaftList - shaft_radius_fu > 0)[0][0]

    if shaft_radius_ox < shaftList[arg_ox-1] + (shaftList[arg_ox] - shaftList[arg_ox-1])/3:
        tp_dict["oxidizer_pump"]["shaft_radius"]= shaftList[arg_ox-1]
    else:
        tp_dict["oxidizer_pump"]["shaft_radius"] = shaftList[arg_ox]

    if shaft_radius_fu < shaftList[arg_fu-1] + (shaftList[arg_fu] - shaftList[arg_fu-1])/3:
        tp_dict["fuel_pump"]["shaft_radius"]= shaftList[arg_fu-1]
    else:
        tp_dict["fuel_pump"]["shaft_radius"] = shaftList[arg_fu]

    return tp_dict

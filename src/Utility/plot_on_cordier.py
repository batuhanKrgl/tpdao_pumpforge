import numpy as np
from matplotlib import pyplot as plt

from src.fluid import Fluid

def cordier_plot(ax, TP_dict:dict, oxi:Fluid, fuel:Fluid):

    rpm_list = np.array(TP_dict["rpm_list"])
    flow_corrector_ox = 1
    flow_corrector_fu = 1

    head_ox = (TP_dict["oxidizer_pump"]["pressure_required"] - TP_dict["oxidizer_pump"][
        "inlet_pressure"]) / 9.805 / oxi.density
    head_fu = (TP_dict["fuel_pump"]["pressure_required"] - TP_dict["fuel_pump"][
        "inlet_pressure"]) / 9.805 / fuel.density

    if TP_dict["oxidizer_pump"]["second_stage"]:
        head_ox *= 0.5
    if TP_dict["fuel_pump"]["second_stage"]:
        head_fu *= 0.5

    if TP_dict["oxidizer_pump"]["double_suction"]:
        flow_corrector_ox = 0.5
    if TP_dict["fuel_pump"]["double_suction"]:
        flow_corrector_fu = 0.5

    specific_speed_ox = rpm_list * np.sqrt(
        TP_dict["oxidizer_pump"]["mass_flow"] * flow_corrector_ox / oxi.density) / np.power(head_ox, 0.75)
    specific_speed_fu = rpm_list * np.sqrt(
        TP_dict["fuel_pump"]["mass_flow"] * flow_corrector_fu / fuel.density) / np.power(head_fu, 0.75)

    pressure_coef_ox = 1.21 * np.exp(-0.77 * specific_speed_ox / 100)
    pressure_coef_fu = 1.21 * np.exp(-0.77 * specific_speed_fu / 100)

    d2_ox = 60 / np.pi / rpm_list * np.sqrt(2 * 9.805 * head_ox / pressure_coef_ox)
    d2_fu = 60 / np.pi / rpm_list * np.sqrt(2 * 9.805 * head_fu / pressure_coef_fu)

    specific_speed_ox *= 2.437742312910564
    specific_speed_fu *= 2.437742312910564

    specific_diameter_ox = 0.7430260979418515 * d2_ox * np.power(head_ox, 0.25) / np.sqrt(
        TP_dict["oxidizer_pump"]["mass_flow"] * flow_corrector_ox / oxi.density)
    specific_diameter_fu = 0.7430260979418515 * d2_fu * np.power(head_fu, 0.25) / np.sqrt(
        TP_dict["fuel_pump"]["mass_flow"] * flow_corrector_fu / fuel.density)


    #fig, ax = plt.subplots()

    # ax.set_yscale("log")
    # ax.set_xscale("log")

    # ax.plot([(np.log10(ns) + 3) / 7 * 4148, (np.log10(ns) + 3) / 7 * 4148, 0],
    #         [1774, 1774 - (np.log10(ds) + 1) / 3 * 1774, 1774 - (np.log10(ds) + 1) / 3 * 1774], ms=2,
    #         lw=0.8, label="Series 1")
    # ax.scatter([(np.log10(ns[0]) + 3) / 7 * 4148, (np.log10(ns[0]) + 3) / 7 * 4148, 0],
    #         [1774, 1774 - (np.log10(ds[0]) + 1) / 3 * 1774, 1774 - (np.log10(ds[0]) + 1) / 3 * 1774], marker="s", ms=2,
    #         )

    ax.plot((np.log10(specific_speed_ox) + 3) / 7 * 4148, 1774 - (np.log10(specific_diameter_ox) + 1) / 3 * 1774, marker="o",
            color="blue", ms=8, lw=1, alpha=0.95,fillstyle="none")
    ax.plot((np.log10(specific_speed_fu) + 3) / 7 * 4148, 1774 - (np.log10(specific_diameter_fu) + 1) / 3 * 1774, marker="1",
            color="red", ms=8, lw=1, alpha=0.95)


def cordier_plot_on(ax, ns, ds, unit="SI", label=None, size="large"):
    if unit == "SI":
        ns *= 2.437742312910564
        ds *= 0.7430260979418515
    marker_list = ["o", "v", "s", "D", "^"]
    if size == "large":
        ax.plot((np.log10(ns) + 3) / 7 * 4148, 1774 - (np.log10(ds) + 1) / 3 * 1774,
                marker=marker_list[len(ax.get_lines())],
                color="C" + str(len(ax.get_lines()) + 1), ms=8, lw=2, alpha=0.95, label=label)
    else:
        ax.plot((np.log10(ns) - 1) / 3 * 1776, 1181 - (np.log10(ds) + 1) / 2 * 1181,
                marker=marker_list[len(ax.get_lines())],
                color="C" + str(len(ax.get_lines()) + 1), ms=8, lw=2, alpha=0.95, label=label)

    plt.show()


def RPM_cordier_plot(rpm, d_p, m_f, Fluid, ax=None):
    g = 9.805
    omega = rpm * np.pi / 30
    v_f = m_f / Fluid.rho
    head = d_p / Fluid.rho / g
    ns = rpm * v_f ** 0.5 / head ** 0.75
    head_coef = 1.21 * np.e ** (-0.408 * ns / 52.9)
    d = 84.6 / (omega * 30 / np.pi) * np.sqrt(head / head_coef) * 1.2
    ds = d * head ** 0.25 / v_f ** 0.5

    if ax == None:
        ax = cordier_plot(ns, ds)
        return ax
    else:
        cordier_plot_on(ax, ns, ds)

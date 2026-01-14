import numpy as np
import bezier
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import fsolve
from src.Pump.pump1D import Impeller, Inducer, Volute, Diffuser, Pump
from src.Utility.geometrical_functions import get_node_normals, get_mag_of_vec
from src.Utility.obj_to_dict import transform_to_dict

nop = 100
pi = np.pi
gravity = 9.805


class Curve(np.ndarray):
    def __new__(cls, array=None, number_of_points=None, *args, **kwargs):
        if number_of_points is None:
            number_of_points = nop
        if array is None:
            array = np.zeros((number_of_points, 3))
        obj = np.asarray(array).view(cls)
        obj.bezier = None
        return obj

    def __init__(self, array=None, number_of_points=None):
        if number_of_points is None:
            number_of_points = nop
        if array is None:
            array = np.zeros((number_of_points, 3))
        self.bezier = None

    def __array_finalize__(self, obj):
        if obj is None: return
        self.bezier = getattr(obj, "bezier", "")

    def set_bezier(self, control_points: np.ndarray):
        if control_points.shape[0] == 2:
            control_points = np.concatenate(
                ([control_points[0]], np.zeros((1, control_points.shape[1])), [control_points[1]]))
        self.bezier = bezier.Curve(control_points, degree=control_points.shape[1] - 1)
        t_samp = np.linspace(0, 1, self.shape[0])
        points = self.bezier.evaluate_multi(t_samp)
        self.x = points[0]
        self.y = points[1]
        self.z = points[2]

    def slice_from_curve(self, curve, start_loc, end_loc=1, number=100):
        self.bezier = curve.bezier.specialize(start_loc, end_loc)
        t_samp = np.linspace(0, 1, number)
        points = self.bezier.evaluate_multi(t_samp)
        self.x = points[0]
        self.y = points[1]
        self.z = points[2]

    @property
    def theta(self):
        return np.arctan2(self[:, 1], self[:, 0])

    @theta.setter
    def theta(self, value):
        r = self.r
        if type(value) is not tuple:
            self[:, 0] = r * np.cos(value)
            self[:, 1] = r * np.sin(value)
        else:
            self[value[0], 0] = r * np.cos(value[1])
            self[value[0], 1] = r * np.sin(value[1])

    @property
    def r(self):
        return np.sqrt(self[:, 0] ** 2 + self[:, 1] ** 2)

    @r.setter
    def r(self, value):
        if type(value) is not tuple:
            self[:, 0] = value * np.cos(self.theta)
            self[:, 1] = value * np.sin(self.theta)
        else:
            self[value[0], 0] = value[1] * np.cos(self.theta)
            self[value[0], 1] = value[1] * np.sin(self.theta)

    @property
    def x(self):
        return self[:, 0]

    @x.setter
    def x(self, value):
        if type(value) is not tuple:
            self[:, 0] = value
        else:
            self[value[0], 0] = value[1]

    @property
    def y(self):
        return self[:, 1]

    @y.setter
    def y(self, value):
        if type(value) is not tuple:
            self[:, 1] = value
        else:
            self[value[0], 1] = value[1]

    @property
    def z(self):
        return self[:, 2]

    @z.setter
    def z(self, value):
        if type(value) is not tuple:
            self[:, 2] = value
        else:
            self[value[0], 2] = value[1]

    def to_ndarray(self):
        return self.obj


class CurveSet:
    def __init__(self, guides: int):
        self.hub = Curve()
        self.tip = Curve()
        self.guides = []
        for i in range(guides):
            self.guides.append(Curve())


class Rotor3D:
    def __init__(self, number_of_guides: int = 10):
        self.type = None
        self.hub = Curve()
        self.tip = Curve()
        self.leading_edge = Curve()
        self.trailing_edge = Curve()
        self.blade_hub = Curve()
        self.blade_tip = Curve()
        self.leading_edge_dict = {}
        self.trailing_edge_dict = {}
        self.export_dict = {}
        self.hub_offsets = []
        self.tip_offsets = []
        self.guides_dict = {
            "number_of_guides": number_of_guides,
            "guides": [],
            "blade_guides": []
        }
        self.meridional_dict = {
            "hub_inlet_CP": 0.5,
            "hub_outlet_CP": 0.5,
            "tip_inlet_CP": 0.5,
            "tip_outlet_CP": 0.5,
            "leading_edge_@hub": 0.2,
            "leading_edge_@tip": 0.2,
            "trailing_edge_@hub": 1,
            "trailing_edge_@tip": 1,
            "leading_edge_@guide": [2, 0.1],
            "inlet_beta_dist": "lineer",
            "outlet_beta_dist": "lineer"
        }
        self.beta_dict = {
            "array": np.zeros((self.guides_dict["number_of_guides"] + 2, nop - 1)),
            "curves": [],
            "hub_beta_CP": [[0.2], [0.8]],
            "tip_beta_CP": [[0.2], [0.8]],
            "inlet_betas": {
                "method": "linear",
                "array": np.zeros(self.guides_dict["number_of_guides"] + 2),
            },
            "outlet_betas": {
                "method": "linear",
                "array": np.zeros(self.guides_dict["number_of_guides"] + 2),
            },
        }
        self.thickness_dict = {
            "hub": {
                "number_of_piece": 1,
                "nodes": [0, 1],
                "thickness": np.array([0.002, 0.002]),
                "distribution": np.array([0.5, 0.5]),
                "location": np.array([0, 1])
            },
            "tip": {
                "number_of_piece": 1,
                "nodes": [0, 1],
                "thickness": np.array([0.002, 0.002]),
                "distribution": np.array([0.5, 0.5]),
                "location": np.array([0, 1])
            }
        }
        self.foil_dict = {
            "pressure": {
                "method": "linear",
                "thickness_array": np.zeros((self.guides_dict["number_of_guides"] + 2, nop)),
                "curves": []
            },
            "suction": {
                "method": "linear",
                "thickness_array": np.zeros((self.guides_dict["number_of_guides"] + 2, nop)),
                "curves": []
            }
        }

    @property
    def outlet_width(self):
        return self.hub.z[-1] - self.tip.z[-1]

    def configure_guides(self, number_of_guides: int) -> None:
        self.guides_dict["number_of_guides"] = number_of_guides
        self.guides_dict["guides"] = []
        self.guides_dict["blade_guides"] = []
        self.beta_dict["array"] = np.zeros((number_of_guides + 2, nop - 1))
        self.beta_dict["inlet_betas"]["array"] = np.zeros(number_of_guides + 2)
        self.beta_dict["outlet_betas"]["array"] = np.zeros(number_of_guides + 2)
        self.foil_dict["suction"]["thickness_array"] = np.zeros((number_of_guides + 2, nop))
        self.foil_dict["pressure"]["thickness_array"] = np.zeros((number_of_guides + 2, nop))
        self.foil_dict["suction"]["curves"] = []
        self.foil_dict["pressure"]["curves"] = []

    @property
    def width(self):
        return self.hub.z[-1] - self.hub.z[0]

    @property
    def diameter(self):
        return self.hub.r[-1] * 2

    @property
    def eye_diameter(self):
        return self.tip.r[0] * 2

    def get_bezier_nodes_from_meridional_dict(self, curve):
        return np.zeros(2)

    def set_meridional_geometry(self, category="all"):
        if category in ["all", "hub"]:
            self.hub.set_bezier(self.get_bezier_nodes_from_meridional_dict("hub"))

        if category in ["all", "tip"]:
            self.tip.set_bezier(self.get_bezier_nodes_from_meridional_dict("tip"))

        if category in ["all", "guide"]:
            self.guides_dict["guides"] = []
            for i in range(self.guides_dict["number_of_guides"]):
                step = (i + 1) / (self.guides_dict["number_of_guides"] + 1)
                curve = Curve()
                curve.set_bezier(self.hub.bezier.nodes * (1 - step) + self.tip.bezier.nodes * step)
                self.guides_dict["guides"].append(curve)

        if category in ["all", "leading"]:
            leading_edge_nodes = np.concatenate((self.hub.bezier.evaluate(self.meridional_dict["leading_edge_@hub"]),
                                                 self.guides_dict["guides"][
                                                     self.meridional_dict["leading_edge_@guide"][0]].bezier.evaluate(
                                                     self.meridional_dict["leading_edge_@guide"][1]),
                                                 self.tip.bezier.evaluate(self.meridional_dict["leading_edge_@tip"])),
                                                axis=1)
            self.leading_edge.set_bezier(leading_edge_nodes)

        if category in ["all", "trailing"]:
            trailing_edge_nodes = np.concatenate((self.hub.bezier.evaluate(self.meridional_dict["trailing_edge_@hub"]),
                                                  self.tip.bezier.evaluate(self.meridional_dict["trailing_edge_@tip"])),
                                                 axis=1)
            self.trailing_edge.set_bezier(trailing_edge_nodes)

    def set_blade_curves(self):
        self.blade_hub.slice_from_curve(self.hub, self.meridional_dict["leading_edge_@hub"],
                                        self.meridional_dict["trailing_edge_@hub"])
        self.blade_tip.slice_from_curve(self.tip, self.meridional_dict["leading_edge_@tip"],
                                        self.meridional_dict["trailing_edge_@tip"])
        self.guides_dict["blade_guides"] = []
        for i, guide in enumerate(self.guides_dict["guides"]):
            nodes = guide.bezier.nodes
            curve_2d = bezier.Curve(np.delete(nodes, 1, axis=0), degree=3)
            nodes_le = self.leading_edge.bezier.nodes
            curve_2d_le = bezier.Curve(np.delete(nodes_le, 1, axis=0), degree=2)
            intersection = curve_2d.intersect(curve_2d_le)
            curve = Curve()
            curve.slice_from_curve(guide, intersection[0, 0], np.linspace(
                self.meridional_dict["trailing_edge_@hub"],
                self.meridional_dict["trailing_edge_@tip"],
                len(self.guides_dict["guides"]) + 2)[i + 1])
            self.guides_dict["blade_guides"].append(curve)

    def set_beta_array(self):
        inlet_cps = np.linspace(self.beta_dict["hub_beta_CP"][0], self.beta_dict["tip_beta_CP"][0],
                                self.beta_dict["array"].shape[0])
        outlet_cps = np.linspace(self.beta_dict["hub_beta_CP"][1], self.beta_dict["tip_beta_CP"][1],
                                 self.beta_dict["array"].shape[0])

        self.beta_cps = np.stack((np.zeros_like(inlet_cps), inlet_cps, outlet_cps, np.ones_like(inlet_cps)), axis=2)

        if self.beta_dict["inlet_betas"]["method"] == "linear":
            self.beta_dict["inlet_betas"]["array"] = np.linspace(self.beta_dict["inlet_betas"]["array"][0],
                                                                 self.beta_dict["inlet_betas"]["array"][-1],
                                                                 self.beta_dict["inlet_betas"]["array"].shape[0])
        else:
            pass

        if self.beta_dict["outlet_betas"]["method"] == "linear":
            self.beta_dict["outlet_betas"]["array"] = np.linspace(self.beta_dict["outlet_betas"]["array"][0],
                                                                  self.beta_dict["outlet_betas"]["array"][-1],
                                                                  self.beta_dict["outlet_betas"]["array"].shape[0])
        else:
            pass
        for i in range(self.beta_dict["array"].shape[0]):
            curve = (bezier.Curve(self.beta_cps[i], degree=3))
            t_samp = np.linspace(0, 1, self.beta_dict["array"].shape[1])
            points = curve.evaluate_multi(t_samp)[0]
            multiplier = self.beta_dict["outlet_betas"]["array"][i] - self.beta_dict["inlet_betas"]["array"][i]
            self.beta_dict["array"][i] = self.beta_dict["inlet_betas"]["array"][i] + multiplier * points

    def set_thetas_from_betas(self):
        self.set_blade_curves()

        def equation(t3, x1, z1, x2, z2, angle_rad):
            y1 = np.zeros_like(x1)
            y2 = np.zeros_like(x2)
            r2 = np.sqrt(x2 ** 2 + y2 ** 2)
            z3 = z2
            x3 = r2 * np.cos(t3)
            y3 = r2 * np.sin(t3)

            vec1 = np.array([x2 - x1, y2 - y1, z2 - z1])
            vec2 = np.array([x3 - x1, y3 - y1, z3 - z1])

            angle = np.arccos(sum(v1 * v2 for v1, v2 in zip(vec1, vec2)) / (
                    np.sqrt(sum(v ** 2 for v in vec1)) * np.sqrt(sum(v ** 2 for v in vec2))))

            return angle - angle_rad

        for curve, beta_list in zip([self.blade_hub] + self.guides_dict["blade_guides"] + [self.blade_tip],
                                    self.beta_dict["array"]):
            theta3 = fsolve(equation, np.ones_like(beta_list) * 1 / 180 * np.pi, args=(curve.r[:-1],
                                                                                       curve.z[:-1],
                                                                                       curve.r[1:],
                                                                                       curve.z[1:],
                                                                                       0.5 * np.pi - beta_list))
            theta3 = np.cumsum(np.abs(theta3))
            curve.theta = np.concatenate(([0], theta3))
        for curve in [self.blade_hub] + self.guides_dict["blade_guides"] + [self.blade_tip]:
            curve.theta += self.blade_hub.theta[-1] - curve.theta[-1]

    def set_thickness_array(self):
        hub_dict = self.thickness_dict["hub"]
        hub_pressure_array = np.linspace(0, 1, nop)
        hub_suction_array = np.linspace(0, 1, nop)
        for i in range(hub_dict["number_of_piece"]):
            start_ind = int(nop * hub_dict["location"][i])
            end_ind = int(nop * hub_dict["location"][i + 1])
            hub_pressure_array[start_ind:end_ind] = np.linspace(hub_dict["thickness"][i] * hub_dict["distribution"][i],
                                                                hub_dict["thickness"][i + 1] * hub_dict["distribution"][
                                                                    i],
                                                                len(hub_pressure_array[start_ind:end_ind]))
            hub_suction_array[start_ind:end_ind] = np.linspace(
                hub_dict["thickness"][i] * (1 - hub_dict["distribution"][i]),
                hub_dict["thickness"][i + 1] * (1 - hub_dict["distribution"][
                    i]),
                len(hub_suction_array[start_ind:end_ind]))

        tip_dict = self.thickness_dict["tip"]
        tip_pressure_array = np.linspace(0, 1, nop)
        tip_suction_array = np.linspace(0, 1, nop)
        for i in range(tip_dict["number_of_piece"]):
            start_ind = int(nop * tip_dict["location"][i])
            end_ind = int(nop * tip_dict["location"][i + 1])
            tip_pressure_array[start_ind:end_ind] = np.linspace(tip_dict["thickness"][i] * tip_dict["distribution"][i],
                                                                tip_dict["thickness"][i + 1] * tip_dict["distribution"][
                                                                    i],
                                                                len(tip_pressure_array[start_ind:end_ind]))
            tip_suction_array[start_ind:end_ind] = np.linspace(
                tip_dict["thickness"][i] * (1 - tip_dict["distribution"][i]),
                tip_dict["thickness"][i + 1] * (1 - tip_dict["distribution"][
                    i]),
                len(tip_suction_array[start_ind:end_ind]))

        self.foil_dict["suction"]["thickness_array"] = np.linspace(hub_suction_array, tip_suction_array,
                                                                   self.foil_dict["suction"]["thickness_array"].shape[
                                                                       0])
        self.foil_dict["pressure"]["thickness_array"] = np.linspace(hub_pressure_array, tip_pressure_array,
                                                                    self.foil_dict["pressure"]["thickness_array"].shape[
                                                                        0])

    def set_foils(self):

        surface_mesh = np.zeros((self.guides_dict["number_of_guides"] + 2, nop, 3))
        for i, blade_line in enumerate([self.blade_hub] + self.guides_dict["blade_guides"] + [self.blade_tip]):
            surface_mesh[i, :, 0] = blade_line.x
            surface_mesh[i, :, 1] = blade_line.y
            surface_mesh[i, :, 2] = blade_line.z

        node_normals = get_node_normals(surface_mesh)
        suction_thickness = np.zeros_like(node_normals)
        pressure_thickness = np.zeros_like(node_normals)

        suction_thickness[:, :, 0] = self.foil_dict["suction"]["thickness_array"]
        suction_thickness[:, :, 1] = self.foil_dict["suction"]["thickness_array"]
        suction_thickness[:, :, 2] = self.foil_dict["suction"]["thickness_array"]

        pressure_thickness[:, :, 0] = self.foil_dict["pressure"]["thickness_array"]
        pressure_thickness[:, :, 1] = self.foil_dict["pressure"]["thickness_array"]
        pressure_thickness[:, :, 2] = self.foil_dict["pressure"]["thickness_array"]

        pressure_surface = surface_mesh + pressure_thickness * node_normals
        suction_surface = surface_mesh - suction_thickness * node_normals

        self.foil_dict["pressure"]["curves"] = []
        self.foil_dict["suction"]["curves"] = []
        for i in range(pressure_surface.shape[0]):
            self.foil_dict["pressure"]["curves"].append(Curve(pressure_surface[i]))
            self.foil_dict["suction"]["curves"].append(Curve(suction_surface[i]))

    def set_leading_edge(self, num=20):
        self.leading_edge_dict["curves"] = []
        for i in range(self.guides_dict["number_of_guides"] + 2):
            self.leading_edge_dict["curves"].append(Curve(number_of_points=num))

        if self.leading_edge_dict["method"] == "Eliptik":
            for i, camber in enumerate([self.blade_hub] + self.guides_dict["blade_guides"] + [self.blade_tip]):
                c = (self.foil_dict["pressure"]["curves"][i][0] +
                     self.foil_dict["suction"]["curves"][i][0]) / 2
                eu = - (camber[0] - camber[1]) / get_mag_of_vec(
                    camber[0] - camber[1])
                u = self.leading_edge_dict["ratio"] * (self.foil_dict["pressure"]["thickness_array"][i, 0] +
                                                       self.foil_dict["suction"]["thickness_array"][i, 0]) * eu
                v = self.foil_dict["pressure"]["curves"][i][0] - c
                self.leading_edge_dict["curves"][i] = c + np.cos(
                    np.pi * np.linspace(0.5, 1.5, num).reshape(num, 1)) * u.reshape(1, 3) + np.sin(
                    np.pi * np.linspace(0.5, 1.5, num).reshape(num, 1)) * v.reshape(1, 3)

    def set_trailing_edge(self, num=20):
        self.trailing_edge_dict["curves"] = []
        for i in range(self.guides_dict["number_of_guides"] + 2):
            self.trailing_edge_dict["curves"].append(Curve(number_of_points=num))

        if self.trailing_edge_dict["method"] == "Çıkış Üzerinde":
            for i, camber in enumerate([self.blade_hub] + self.guides_dict["blade_guides"] + [self.blade_tip]):
                vec = self.foil_dict["suction"]["curves"][i][-1] - \
                      self.foil_dict["suction"]["curves"][i][-2]
                e = vec / np.sqrt(np.sum(np.square(vec)))
                point_p = self.foil_dict["pressure"]["curves"][i][-1]
                point_s = self.foil_dict["suction"]["curves"][i][-1]
                thickness = np.sqrt(np.sum(np.square(point_p - point_s)))
                point_se = point_s + e * thickness * 1.5
                self.trailing_edge_dict["curves"][i] = Curve(array=np.linspace(point_p, point_se, num))

        if self.trailing_edge_dict["method"] == "Eliptik":
            for i, camber in enumerate([self.blade_hub] + self.guides_dict["blade_guides"] + [self.blade_tip]):
                c = (self.foil_dict["pressure"]["curves"][i][-1] +
                     self.foil_dict["suction"]["curves"][i][-1]) / 2
                eu = - (camber[-1] - camber[-2]) / get_mag_of_vec(
                    camber[-1] - camber[-2])
                u = self.trailing_edge_dict["ratio"] * (self.foil_dict["pressure"]["thickness_array"][i, -1] +
                                                        self.foil_dict["suction"]["thickness_array"][i, -1]) * eu
                v = self.foil_dict["pressure"]["curves"][i][-1] - c
                self.trailing_edge_dict["curves"][i] = Curve(
                    np.cos(np.pi * np.linspace(0.5, 1.5, num).reshape(num, 1)) * u.reshape(1, 3) +
                    np.sin(np.pi * np.linspace(0.5, 1.5, num).reshape(num, 1)) * v.reshape(1, 3) + c)

    def export_as_dict(self):
        self.export_dict = transform_to_dict(self)

    def export_npy(self, path="temp/"):
        np.save(path + "hub.npy", self.hub.obj)
        np.save(path + "tip.npy", self.tip.obj)
        np.save(path + "lea.npy", self.leading_edge.obj)
        np.save(path + "tra.npy", self.trailing_edge.obj)

        pressure_side_np = np.zeros((len(self.foil_dict["pressure"]["curves"]), nop, 3))
        suction_side_np = np.zeros_like(pressure_side_np)
        leading_np = np.zeros((len(self.leading_edge_dict["curves"]), 20, 3))
        trailing_np = np.zeros_like(leading_np)

        for i in range(pressure_side_np.shape[0]):
            pressure_side_np[i] = self.foil_dict["pressure"]["curves"][i].obj
            suction_side_np[i] = self.foil_dict["suction"]["curves"][i].obj
            leading_np[i] = self.leading_edge_dict["curves"][i].obj
            trailing_np[i] = self.trailing_edge_dict["curves"][i].obj

        np.save(path + "pressure.npy", pressure_side_np)
        np.save(path + "suction.npy", suction_side_np)
        np.save(path + "leading.npy", leading_np)
        np.save(path + "trailing.npy", trailing_np)


class Impeller3D(Rotor3D):
    def __init__(self, impeller1D: Impeller):
        super().__init__()
        self.type = "impeller"
        self.meridional_dict["hub_inlet_radius"] = impeller1D.hub.inlet.radius
        self.meridional_dict["tip_inlet_radius"] = impeller1D.tip.inlet.radius
        self.meridional_dict["hub_outlet_radius"] = impeller1D.hub.outlet.radius
        self.meridional_dict["outlet_width"] = impeller1D.outlet_width
        self.meridional_dict["width"] = impeller1D.width
        self.meridional_dict["trailing_edge_@hub"] = 1.0
        self.meridional_dict["trailing_edge_@tip"] = 1.0

        self.beta_dict["inlet_betas"]["array"][0] = impeller1D.hub.inlet.beta_blade
        self.beta_dict["inlet_betas"]["array"][-1] = impeller1D.tip.inlet.beta_blade
        self.beta_dict["outlet_betas"]["array"][0] = impeller1D.hub.outlet.beta_blade
        self.beta_dict["outlet_betas"]["array"][-1] = impeller1D.tip.outlet.beta_blade

        self.leading_edge_dict = {
            "method": "Eliptik",
            "ratio": 3,
            "curves": []
        }
        self.trailing_edge_dict = {
            "method": "Çıkış Üzerinde"
        }

    def initialize(self):
        self.set_meridional_geometry()
        self.set_beta_array()
        self.set_thetas_from_betas()
        self.set_thickness_array()
        self.set_foils()
        self.set_leading_edge()
        self.set_trailing_edge()

    def get_bezier_nodes_from_meridional_dict(self, curve):
        merid_dict = self.meridional_dict
        if curve == "hub":
            curve_axial = merid_dict["width"]
        else:
            curve_axial = merid_dict["width"] - merid_dict["outlet_width"]
        nodes = np.array([[
            merid_dict[f"{curve}_inlet_radius"],
            merid_dict[f"{curve}_inlet_radius"],
            merid_dict[f"{curve}_inlet_radius"] + merid_dict[f"{curve}_outlet_CP"] * (
                    merid_dict["hub_outlet_radius"] - merid_dict[f"{curve}_inlet_radius"]),
            merid_dict["hub_outlet_radius"]],
            [
                0,
                merid_dict[f"{curve}_inlet_CP"] * curve_axial,
                curve_axial,
                curve_axial]])
        return nodes


class Inducer3D(Rotor3D):
    def __init__(self, inducer1D: Inducer, number_of_guides: int = 10):
        super().__init__(number_of_guides=number_of_guides)
        self.type = "inducer"
        self.meridional_dict["hub_inlet_radius"] = inducer1D.hub.inlet.radius
        self.meridional_dict["tip_inlet_radius"] = inducer1D.tip.inlet.radius
        self.meridional_dict["hub_outlet_radius"] = inducer1D.hub.outlet.radius
        self.meridional_dict["tip_outlet_radius"] = inducer1D.tip.outlet.radius
        self.meridional_dict["width"] = inducer1D.width * 1.2
        self.meridional_dict["trailing_edge_@hub"] = 0.95
        self.meridional_dict["trailing_edge_@tip"] = 0.95

        self.beta_dict["inlet_betas"]["array"][0] = inducer1D.hub.inlet.beta
        self.beta_dict["inlet_betas"]["array"][-1] = inducer1D.tip.inlet.beta
        self.beta_dict["outlet_betas"]["array"][0] = inducer1D.hub.outlet.beta
        self.beta_dict["outlet_betas"]["array"][-1] = inducer1D.tip.outlet.beta

        self.trailing_edge_dict = {
            "method": "Eliptik",
            "ratio": 2,
            "curves": []
        }
        self.leading_edge_dict = {
            "method": "Eliptik",
            "ratio": 1,
            "curves": []
        }

    def initialize(self):
        self.set_meridional_geometry()
        self.set_beta_array()
        self.set_thetas_from_betas()
        self.set_thickness_array()
        self.set_foils()
        self.set_leading_edge()
        self.set_trailing_edge()
        # self.set_foil_offsets()

    def get_bezier_nodes_from_meridional_dict(self, curve):
        merid_dict = self.meridional_dict
        nodes = np.array([[
            merid_dict[f"{curve}_inlet_radius"],
            merid_dict[f"{curve}_inlet_radius"],
            merid_dict[f"{curve}_outlet_radius"],
            merid_dict[f"{curve}_outlet_radius"]],
            [
                0,
                merid_dict[f"{curve}_inlet_CP"] * merid_dict["width"],
                merid_dict[f"{curve}_outlet_CP"] * merid_dict["width"],
                merid_dict["width"]]])
        return nodes


class Volute3D:
    #  SI birimlerinde [m, s, Pa, rad], Impeller gereklidir.
    #  30 derecelik konik simetrik kesit ile salyangoz kayıpları hesaplar.
    #  TODO: Kesit tipleri eklenecektir.
    def __init__(self, volute1D: Volute, impeller3D: Impeller3D):
        #       fluid: Akışkan, Fluid,
        #       impeller: Çark, Impeller
        #       vol_flow: Hacimsel Debi [m3/s],
        #       outlet_radius: Pompa Çıkış Yarıçapı [m],
        #       is_double_suction: Çark emiş adet kontrolu, True: Çift Emiş, False: Tek Emiş,
        #       clearance: Çark-Salyangoz arası açıklık [m],
        #       roughness: Salyangoz yüzey pürüzlülüğü [m]
        self.use_imp_inlet_flow = True
        self.volute1D = volute1D
        self.impeller1D = self.volute1D.impeller
        self.impeller3D = impeller3D
        self.fluid = self.volute1D.fluid  # Kullanılan akışkan
        self.is_double_suction = self.volute1D.is_double_suction

        self.impeller_disk_thickness = 2e-3
        self.impeller_width = self.impeller3D.outlet_width * 2 + self.impeller_disk_thickness if self.is_double_suction else self.impeller3D.outlet_width
        self.anchor_z = self.impeller3D.tip.z[-1] + 0.5 * self.impeller_width
        self.anchor_r = self.impeller3D.hub.r[-1]
        self.offset_r = 1e-3
        self.offset_hub_z = 0.5e-3
        self.offset_tip_z = 0.5e-3

        self.roughness = self.volute1D.roughness  # Metal pürüzlülüğü
        self.diffuser_expansion_angle = self.volute1D.diffuser_expansion_angle

        self.cut_water = CutWater3D(self)
        self.cut_water.calc_radius()

        outlet_c_u = 0.5 * (self.impeller1D.tip.outlet.c_u + self.impeller1D.tip.outlet.c_u)
        self._inlet_cu = outlet_c_u * self.impeller3D.diameter / self.diameter
        self._ref_vol_flow = self.volute1D.vol_flow  # Volumetrik debi
        self.blind_vol_flow = 0

        self.cross_section = {
            "trapezoid": {
                "trapezoid_angle": 30 * np.pi / 180,
            },
            "radius_based": {
                "cone_angle": 30 * np.pi / 180,
                "base_height": 2e-3,
                "main_radius": 50e-3,
                "base_radius": 2e-3,
                "corner_radius": 5e-3,
            }
        }
        self.cross_section_type = "trapezoid"
        self.wrap_angle = 360
        self.angular_locs = np.linspace(0, self.wrap_angle, 360 + 1)  # Hangi açısal kesitlerde hesap yapılsın.
        self.sections = []
        self.section_areas = np.zeros_like(self.angular_locs)  # Kesit alanları
        self.section_wet_area = np.zeros_like(self.angular_locs)  # Kesit için dilim kalınlığındaki ıslak yüzey alanı
        self.section_lengths = np.zeros_like(self.angular_locs)
        self.section_velocity = np.zeros_like(self.angular_locs)  # Kesit ortalama hızı

        self.friction_head_loss = 0
        self.total_head_loss = 0
        self.shock_head_loss = 0

        self.calc_sections()

        self.exit_diff = ExitDiffuser3D(self)

        self.calc_shock_head_loss()
        self.calc_friction_loss()
        self.calc_total_loss()

    def export_dict(self):
        input_dict = {
            "impeller_disk_thickness": self.impeller_disk_thickness,
            "impeller_width": self.impeller_width,
            "anchor_z": self.anchor_z,
            "anchor_r": self.anchor_r,
            "offset_r": self.offset_r,
            "offset_hub_z": self.offset_hub_z,
            "offset_tip_z": self.offset_tip_z,
            "roughness": self.roughness,
            "diffuser_expansion_angle": self.diffuser_expansion_angle,
            "_inlet_cu": self._inlet_cu,
            "_ref_vol_flow": self._ref_vol_flow,
            "blind_vol_flow": self.blind_vol_flow,
            "cross_section": self.cross_section,
            "cross_section_type": self.cross_section_type,
            "wrap_angle": self.wrap_angle,
            "friction_head_loss": self.friction_head_loss,
            "total_head_loss": self.total_head_loss,
            "shock_head_loss": self.shock_head_loss,
            "exit_diff": {
                "outlet_radius": self.exit_diff.outlet_radius,
                "height": self.exit_diff.height,
                "eccentricity": self.exit_diff.eccentricity,
                "eccentricity_mode": self.exit_diff.eccentricity_mode,
                "z_offset": self.exit_diff.z_offset,
                "num_sections": self.exit_diff.num_sections,
            },
            "cut_water": {
                "radius": self.cut_water.radius,
                "thickness": self.cut_water.thickness,
                "volute_radius": self.cut_water.volute_radius,
                "sample_count": self.cut_water.sample_count,
                "start_angle": self.cut_water.start_angle,
                "edge_volute_t_from": self.cut_water.edge_volute_t_from,
                "edge_volute_t_to": self.cut_water.edge_volute_t_to,
                "bezier_top_t_volute": self.cut_water.bezier_top_t_volute,
                "bezier_top_t_diffuser": self.cut_water.bezier_top_t_diffuser,
                "bezier_bottom_t_volute": self.cut_water.bezier_bottom_t_volute,
                "bezier_bottom_t_diffuser": self.cut_water.bezier_bottom_t_diffuser,
                "section_numb": self.cut_water.section_numb,
                "patch_top_diffuser": self.cut_water.patch_top_diffuser,
                "patch_top_volute": self.cut_water.patch_top_volute,
                "patch_bottom_diffuser": self.cut_water.patch_bottom_diffuser,
                "patch_bottom_volute": self.cut_water.patch_bottom_volute,
            },
        }
        return input_dict

    def import_dict(self, import_dict):
        for key in import_dict:
            if key in ["cut_water", "exit_diff"]:
                for inner_key in import_dict[key]:
                    setattr(getattr(self, key), inner_key, import_dict[key][inner_key])
            else:
                setattr(self, key, import_dict[key])
        self.set_wrap_angle()
        self.calc_sections()
        self.exit_diff.update_surface()

    def set_wrap_angle(self):
        self.sections = []
        self.section_areas = np.zeros_like(self.angular_locs)  # Kesit alanları
        self.section_wet_area = np.zeros_like(self.angular_locs)  # Kesit için dilim kalınlığındaki ıslak yüzey alanı
        self.section_lengths = np.zeros_like(self.angular_locs)
        self.section_velocity = np.zeros_like(self.angular_locs)  # Kesit ortalama hızı
        self.calc_sections()

    def calc_sections(self, method="trapezoid"):
        if method == "trapezoid":
            trapezoid_angle = self.cross_section["trapezoid"]["trapezoid_angle"]

            def calc_epsilon(c_l, e):
                # Scipy.fsolve için hazırlanmıştır.
                # Salyangozun hangi açısal konumunda, kesitin konik çizgi uzunluğunun ne kadar olacağını hesaplar.
                # Gülich Eq. 7.24' ü kullanır.
                # "height" ve "integral"in değiştirilmesi ile farklı kesit şekillerine uyarlanabilir.
                # Simetriklikten ötürü yarım kesit hesaplanır. Gerekli yerlerde iki ile çarpılır.
                r_start = self.inlet_radius
                r_end = self.inlet_radius + c_l * np.cos(trapezoid_angle)
                radius, r_step = np.linspace(r_start, r_end, 100 - 1, retstep=True)
                section_slice_centers = (radius[1:] + r_step / 2).transpose()  # integral dilimlerinin merkezleri
                height = (section_slice_centers - r_start) * np.tan(trapezoid_angle) + 0.5 * self.inlet_width
                self.section_velocity = np.sum(  # Alan ortalama
                    self.ref_inlet_cu * self.inlet_radius / section_slice_centers * (
                            height.transpose() * r_step).transpose(),
                    axis=1) / np.sum((height.transpose() * r_step).transpose(), axis=1)

                integral = np.tan(trapezoid_angle) * (  # Kesite özgü analitik integral çözümü
                        (r_end - r_start) - r_start * np.log(r_end / r_start)) + 0.5 * self.inlet_width * np.log(
                    r_end / r_start)
                epsilon_calc = self.ref_inlet_cu * self.inlet_radius / self._ref_vol_flow * 360 * 2 * integral
                return epsilon_calc - e

            # Salyangozun açısal konumu ile kesit büyüklüklerini eşleyen numerik çözüm.
            # Farklı kesit tipleri için kesit alanını kontrol eden bir parametre kurgulanmalı. Burada koninin çizgi uzunluğu
            # Başlangıç değeri, giriş yüksekliğinden türetilmiştir. (mertebe tutturmak yeterli)
            cone_lengths = fsolve(calc_epsilon,
                                  np.linspace((self.cut_water.radius - self.inlet_radius) / np.cos(trapezoid_angle),
                                              self.inlet_width,
                                              self.angular_locs.shape[0]),
                                  args=self.angular_locs)

            # Kesitlerin r_max noktası koordinatları
            tip_points = np.array([self.inlet_radius + cone_lengths * np.cos(trapezoid_angle),
                                   self.inlet_width * 0.5 + cone_lengths * np.sin(trapezoid_angle)])

            self.sections = []
            for i in range(len(cone_lengths)):
                c = Curve()
                c.r = np.linspace(self.inlet_radius, tip_points[0, i], 100)
                c.z = (c.r - self.inlet_radius) * np.tan(trapezoid_angle) + 0.5 * self.inlet_width
                c.theta = np.deg2rad(self.angular_locs[i])

                c2 = Curve()
                c2.r = c.r[-1]
                c2.z = np.linspace(c.z[-1], -c.z[-1], 100)
                c2.theta = np.deg2rad(self.angular_locs[i])

                c3 = Curve()
                c3.r = np.flip(c.r)
                c3.z = -np.flip(c.z)
                c3.theta = c.theta

                self.sections.append([c, c2, c3])
                self.section_areas[i] = (c.z[-1] + 0.5 * self.inlet_width) * (c.r[-1] - self.inlet_radius)
                self.section_lengths[i] = 2 * (cone_lengths[i] + c.z[-1])

            # self.sections[:, :-1, 0] = np.linspace(self.inlet_radius, tip_points[0], 99).transpose()
            # self.sections[:, -1, 0] = self.sections[:, -2, 0]
            # self.sections[:, :-1, 2] = ((self.sections[:, :-1, 0] - self.inlet_radius) *
            #                             np.tan(trapezoid_angle) + 0.5 * self.inlet_width)
            # self.sections[:, -1, 2] = 0
            # self.section_areas = ((self.sections[:, -2, 2] + 0.5 * self.inlet_width) *
            #                       (self.sections[:, -2, 0] - self.inlet_radius))
            # self.section_lengths = 2 * (cone_lengths + self.sections[:, -2, 2])  # Kesitin çizgi uzunluğu

    def calc_friction_loss(self):
        #  Gülich Table 3.8(2), Eq. 3.8.21' göre hesaplanmış sürtünme katsayısı.
        #  Sadece salyangoz için hesaplanmaktadır.
        # TODO: Ortalama hız yerine her section için ayrı hız ve sürtünme katsayısı ile hesaplanacak.
        mean_profile_lengths = 0.5 * (self.section_lengths[1:] + self.section_lengths[:-1])
        profile_centers = np.array([sec[0].r[-1] - (sec[0].r[-1] - sec[0].r[0]) / 3 for sec in self.sections])
        angular_step = self.angular_locs[-1] - self.angular_locs[-2]
        arc_lengths = 0.5 * (profile_centers[1:] + profile_centers[:-1]) * angular_step * np.pi / 180
        self.section_wet_area = mean_profile_lengths * arc_lengths
        reynolds_volute = (0.5 * (self.section_velocity[1:] + self.section_velocity[:-1]) *
                           1.5 * arc_lengths.sum() / self.fluid.kinematicViscosity)
        c_f = 0.0015
        friction_coeff = 0.136 / (-np.log10(0.2 * self.roughness / 1.5 * arc_lengths.sum() +
                                            12.5 / reynolds_volute)) ** 2.15
        self.friction_head_loss = np.sum((0.5 * (self.section_velocity[1:] + self.section_velocity[:-1])) ** 3 *
                                         (
                                                 friction_coeff + c_f) * self.section_wet_area) / self._ref_vol_flow / gravity / 2

    def calc_shock_head_loss(self):
        # Salyangoz girişindeki şok kaybı hesabı.
        self.shock_head_loss = ((
                                        self.impeller1D.tip.outlet.blockage - self.impeller1D.outlet_width / self.inlet_width) ** 2 *
                                self.impeller1D.tip.outlet.flow_number ** 2) * self.impeller1D.tip.outlet.u ** 2 / gravity / 2
        if self.is_double_suction:
            self.shock_head_loss = ((
                                            self.impeller1D.tip.outlet.blockage - self.impeller1D.outlet_width * 2 / self.inlet_width) ** 2 *
                                    self.impeller1D.tip.outlet.flow_number ** 2) * self.impeller1D.tip.outlet.u ** 2 / gravity / 2

    def calc_total_loss(self):
        #  Toplam kaybı hesaplar.
        self.total_head_loss = self.friction_head_loss + self.shock_head_loss + self.exit_diff.head_loss

    def update_from_impeller(self, imp: Impeller):
        # Çarkta bir değişiklik olması durumunda salyangozu günceller.
        self.impeller1D = imp
        self.cut_water.calc_radius()
        self.calc_sections()
        self.calc_shock_head_loss()
        self.calc_friction_loss()
        self.calc_total_loss()

    @property
    def inlet_radius(self):
        return self.diameter / 2

    @property
    def absolute_r(self):
        return self.anchor_r + self.offset_r

    @absolute_r.setter
    def absolute_r(self, value):
        self.offset_r = value - self.anchor_r

    @property
    def absolute_hub_z(self):
        return self.anchor_z + self.impeller_width * 0.5 + self.offset_hub_z

    @absolute_hub_z.setter
    def absolute_hub_z(self, value):
        self.offset_hub_z = value - self.anchor_z - 0.5 * self.impeller_width

    @property
    def absolute_tip_z(self):
        return self.anchor_z - self.impeller_width * 0.5 - self.offset_tip_z

    @absolute_tip_z.setter
    def absolute_tip_z(self, value):
        self.offset_tip_z = self.anchor_z - value - 0.5 * self.impeller_width

    @property
    def diameter(self):
        return 2 * self.absolute_r

    @diameter.setter
    def diameter(self, value):
        self.absolute_r = 0.5 * value

    @property
    def diameter_ratio(self):
        return self.diameter / self.impeller3D.diameter

    @diameter_ratio.setter
    def diameter_ratio(self, value):
        self.diameter = value * self.impeller3D.diameter

    @property
    def inlet_width(self):
        return self.absolute_hub_z - self.absolute_tip_z

    @inlet_width.setter
    def inlet_width(self, value):
        self.absolute_hub_z = self.absolute_tip_z + value

    @property
    def inlet_width_ratio(self):
        return self.inlet_width / self.impeller_width

    @inlet_width_ratio.setter
    def inlet_width_ratio(self, value):
        self.inlet_width = value * self.impeller_width

    @property
    def inlet_area(self):
        return 2 * np.pi * self.inlet_radius * self.inlet_width

    @property
    def inlet_c(self):
        return np.sqrt(self.ref_inlet_cu ** 2 + self.inlet_cm ** 2)

    @property
    def inlet_alpha(self):
        return np.arctan(self.inlet_cm / self.ref_inlet_cu)

    @property
    def inlet_cm(self):
        outlet_c_m = 0.5 * (self.impeller1D.tip.outlet.c_m + self.impeller1D.tip.outlet.c_m)
        return self.impeller1D.outlet_area * outlet_c_m / self.inlet_area

    @property
    def ref_inlet_cu(self):
        if self.use_imp_inlet_flow:
            outlet_c_u = 0.5 * (self.impeller1D.tip.outlet.c_u + self.impeller1D.tip.outlet.c_u)
            self._inlet_cu = outlet_c_u * self.impeller3D.diameter / self.diameter
            return self._inlet_cu
        else:
            return self._inlet_cu

    @ref_inlet_cu.setter
    def ref_inlet_cu(self, value):
        self._inlet_cu = value

    @property
    def ref_vol_flow(self):
        if self.use_imp_inlet_flow:
            self._ref_vol_flow = self.volute1D.vol_flow
            return self._ref_vol_flow
        else:
            return self._ref_vol_flow

    @ref_vol_flow.setter
    def ref_vol_flow(self, value):
        self._ref_vol_flow = value


class ExitDiffuser3D:
    def __init__(self, parent: Volute3D):
        self.parent = parent
        self.outlet_radius = self.parent.volute1D.exit_diff.outlet_radius
        self.height = self.parent.volute1D.exit_diff.length
        self.eccentricity = 0
        self.eccentricity_mode = 0
        self.z_offset = 0
        self.num_sections = 50
        self.surface = self.update_surface()

    @property
    def input_section(self):
        num_of_curve = len(self.parent.sections[-1]) + 1
        array = np.zeros((num_of_curve * 100 - num_of_curve + 1, 3))
        for i in range(num_of_curve):
            if i == num_of_curve - 1:
                array[i * 100 - i:(i + 1) * 100 - i] = np.linspace(self.parent.sections[-1][-1][-1],
                                                                   self.parent.sections[-1][0][0], 100)
            else:
                array[i * 100 - i:(i + 1) * 100 - i] = self.parent.sections[-1][i]
        return array

    @property
    def outlet_diameter(self):
        return self.outlet_radius * 2

    @outlet_diameter.setter
    def outlet_diameter(self, value):
        self.outlet_radius = value / 2

    @property
    def inlet_c(self):
        return self.parent.section_velocity[-1]

    @property
    def inlet_radius(self):
        return (self.parent.section_areas[-1] / np.pi) ** 0.5

    @property
    def inlet_cm(self):
        return self.parent.ref_vol_flow / self.inlet_radius ** 2 / np.pi

    @property
    def center_line(self):
        input_center = np.array([0.5 * (np.max(self.input_section[:, 0]) + np.min(self.input_section[:, 0])), 0, 0])
        if self.eccentricity_mode == 0:
            output_center = input_center + [0, self.height, self.z_offset]
        elif self.eccentricity_mode == 1:
            eccentricity = np.max(self.input_section[:, 0]) - input_center[0] - self.outlet_radius
            output_center = input_center + [eccentricity, self.height, self.z_offset]
        elif self.eccentricity_mode == -1:
            eccentricity = input_center[0] - np.min(self.input_section[:, 0]) - self.outlet_radius
            output_center = input_center + [-eccentricity, self.height, self.z_offset]
        else:
            output_center = input_center + [self.eccentricity, self.height, self.z_offset]
        return np.vstack((input_center, output_center))

    @property
    def output_circle(self):
        input_angles = np.arctan2(self.input_section[:, 2] - self.center_line[0, 2],
                                  self.input_section[:, 0] - self.center_line[0, 0])

        return np.array([
            self.center_line[1, 0] + self.outlet_radius * np.cos(input_angles),
            self.center_line[1, 1] * np.ones_like(input_angles),
            self.center_line[1, 2] + self.outlet_radius * np.sin(input_angles)

        ]).transpose()

    @property
    def area_ratio(self):
        return self.outlet_radius ** 2 / self.inlet_radius ** 2

    @property
    def optimum_height(self):
        return (self.area_ratio - 1.05) * self.inlet_radius / 0.184

    @property
    def cp(self):
        return 0.36 * (self.height / self.inlet_radius) ** 0.26

    @property
    def static_head_rise(self):
        return self.cp / 2 * self.inlet_c ** 2 / gravity

    @property
    def head_loss(self):
        # Difüzör kaybı Gülich Table 3.8(2) Eq. 3.8.22
        return self.inlet_c ** 2 * (1 - self.cp - 1 / self.area_ratio ** 2) * self.parent.ref_vol_flow / 2 / gravity

    @property
    def outlet_c(self):
        # Difüzör çıkışı hesaplanır.
        return ((self.inlet_c ** 2 / 2 / gravity - self.head_loss - self.static_head_rise) * 2 * gravity) ** 0.5

    @property
    def all_sections(self):
        sections = []
        for t in np.linspace(0, 1, self.num_sections + 2):
            intermediate_section = self.input_section * (1 - t) + self.output_circle * t
            sections.append(intermediate_section)
        return np.stack(sections, axis=0)

    def update_surface(self):
        len_u = self.all_sections.shape[1]
        len_v = self.all_sections.shape[0]
        u = np.linspace(0, 1, len_u)
        v = np.linspace(0, 1, len_v)
        U, V = np.meshgrid(u, v)
        X = self.all_sections[:, :, 0]
        Y = self.all_sections[:, :, 1]
        Z = self.all_sections[:, :, 2]
        return LinearNDInterpolator((U.flatten(), V.flatten()), np.array([X.flatten(), Y.flatten(), Z.flatten()]).T)

    @property
    def section_areas(self):
        area_list = []
        for section in self.all_sections:
            points = section[:2, :].T  # X-Y plane
            area_list.append(0.5 * np.abs(
                np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1))))
        return area_list

    # def calculate_triangle_area(self, v1, v2, v3):
    #     return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    #
    # @property
    # def surface_area(self):
    #     total_area = 0
    #     for i in range(self.all_sections.shape[0] - 1):
    #         section1 = self.all_sections[i]
    #         section2 = self.all_sections[i + 1]
    #         for j in range(section1.shape[1] - 1):
    #             v1 = section1[:, j]
    #             v2 = section1[:, j + 1]
    #             v3 = section2[:, j + 1]
    #             v4 = section2[:, j]
    #
    #             total_area += self.calculate_triangle_area(v1, v2, v3)
    #             total_area += self.calculate_triangle_area(v1, v3, v4)
    #
    #         # Close the surface loop by connecting the last points
    #         v1 = section1[:, -1]
    #         v2 = section1[:, 0]
    #         v3 = section2[:, 0]
    #         v4 = section2[:, -1]
    #
    #         total_area += self.calculate_triangle_area(v1, v2, v3)
    #         total_area += self.calculate_triangle_area(v1, v3, v4)
    #
    #     return total_area

    @property
    def cone_angle(self):
        guide_angles_z = np.arctan2(self.output_circle[:, 2] - self.input_section[:, 2],
                                    self.output_circle[:, 1] - self.input_section[:, 1], )
        return np.max(guide_angles_z), np.min(guide_angles_z)


class CutWater3D:
    #  Sınıf düzenleme amacıyla kuruglanmıştır.
    def __init__(self, parent: Volute3D, thickness=2e-3, ang_loc=40):
        self.parent = parent
        self.radius = 0
        self.thickness = thickness
        self.volute_radius = self.parent.inlet_radius
        self.sample_count = 100
        self.start_angle = 40
        self.edge_volute_t_from = 0.4
        self.edge_volute_t_to = 0.6
        self.bezier_top_t_volute = 4
        self.bezier_top_t_diffuser = 4
        self.bezier_bottom_t_volute = 4
        self.bezier_bottom_t_diffuser = 4
        self.section_numb = 11
        self.patch_top_diffuser = 0.05
        self.patch_top_volute = 0.025
        self.patch_bottom_diffuser = 0.05
        self.patch_bottom_volute = 0.025

    def get_volute_section(self, index=None, t=None, numb=None):
        if numb is None:
            numb = self.sample_count
        if t is None:
            t = [0.5]
        if index is None:
            index = int(self.start_angle)
        if self.parent.cross_section_type == "trapezoid":
            index = int(index)
            t_values = np.linspace(0, 1, numb)
            size = (len(self.parent.sections[index][1].x))
            x_points = np.interp(t_values, np.linspace(0, 1, size), self.parent.sections[index][1].x)
            y_points = np.interp(t_values, np.linspace(0, 1, size), self.parent.sections[index][1].y)
            z_points = np.interp(t_values, np.linspace(0, 1, size), self.parent.sections[index][1].z)
            return np.vstack((x_points, y_points, z_points)).transpose()
        else:
            t_values = np.linspace(t[0], t[1], numb)
            fx = np.concatenate([c.x.tolist() for c in self.parent.sections[index]])
            fy = np.concatenate([c.y.tolist() for c in self.parent.sections[index]])
            fz = np.concatenate([c.z.tolist() for c in self.parent.sections[index]])
            x_points = np.interp(t_values, np.linspace(0, 1, len(fx)), fx)
            y_points = np.interp(t_values, np.linspace(0, 1, len(fx)), fy)
            z_points = np.interp(t_values, np.linspace(0, 1, len(fx)), fz)
            return np.vstack((x_points, y_points, z_points)).transpose()

    @property
    def edge_volute(self):
        return self.get_volute_section(t=[self.bezier_top_t_volute, self.bezier_bottom_t_volute])

    def generate_diffuser_bezier(self, index=None, t=None, numb=None):
        if numb is None:
            numb = self.sample_count
        if t is None:
            t = [0, 1]
        if index is None:
            index = self.start_angle

        t_values = np.linspace(t[0], t[1], numb)
        surface = np.flip(self.parent.exit_diff.all_sections[index, -100:], axis=0)
        t_bases = np.linspace(0, 1, len(surface))
        x_points = np.interp(t_values, t_bases, surface[:, 0])
        y_points = np.interp(t_values, t_bases, surface[:, 1])
        z_points = np.interp(t_values, t_bases, surface[:, 2])
        return np.vstack((x_points, y_points, z_points)).transpose()

    @property
    def edge_diffuser(self):
        y_max = np.max(self.edge_volute[:, 1])
        diff_slice_idx = np.argmin(np.abs(self.parent.exit_diff.all_sections[:, 0, 1] - y_max))
        return self.generate_diffuser_bezier(index=diff_slice_idx)

    @staticmethod
    def bezier_curve(P0, P1, P2, P3):
        t = np.linspace(np.zeros(3), np.ones(3), 100)
        return ((1 - t) ** 3 * np.tile(P0, (100, 1)) +
                3 * (1 - t) ** 2 * t * np.tile(P1, (100, 1)) +
                3 * (1 - t) * t ** 2 * np.tile(P2, (100, 1)) +
                t ** 3 * np.tile(P3, (100, 1)))

    @property
    def bezier_top(self):
        return self.surface[-1]

    @property
    def bezier_bottom(self):
        return self.surface[0]

    @property
    def surface(self):
        volute_section = self.get_volute_section(index=self.start_angle,
                                                 t=[self.edge_volute_t_from, self.edge_volute_t_to],
                                                 numb=self.section_numb)
        volute_section2 = self.get_volute_section(index=self.start_angle + 1,
                                                  t=[self.edge_volute_t_from, self.edge_volute_t_to],
                                                  numb=self.section_numb)
        y_max = np.max(self.edge_volute[:, 1])
        diff_slice_idx = np.argmin(np.abs(self.parent.exit_diff.all_sections[:, 0, 1] - y_max))
        dif_section = self.generate_diffuser_bezier(index=diff_slice_idx, numb=self.section_numb)
        dif_section2 = self.generate_diffuser_bezier(index=diff_slice_idx + 1, numb=self.section_numb)
        bezier_list = []
        t_volute_array = np.linspace(self.bezier_top_t_volute, self.bezier_bottom_t_volute, self.section_numb)
        t_diffuser_array = np.linspace(self.bezier_top_t_diffuser, self.bezier_bottom_t_diffuser, self.section_numb)
        for i in range(self.section_numb):
            cp1 = volute_section[i]
            cp2 = (1 - t_volute_array[i]) * volute_section2[i] + t_volute_array[i] * volute_section[i]
            cp3 = (1 - t_diffuser_array[i]) * dif_section2[i] + t_diffuser_array[i] * dif_section[i]
            cp4 = dif_section[i]
            bezier_list.append(self.bezier_curve(cp1, cp2, cp3, cp4))
        return np.array(bezier_list)

    @property
    def filler_top_curve1(self):
        index = np.array(np.where((self.parent.exit_diff.all_sections == tuple(self.edge_diffuser[0])).all(axis=2)))[
                :, 0]
        p1 = np.array([index[1] / (self.parent.exit_diff.all_sections.shape[1] - 1),
                       index[0] / (self.parent.exit_diff.all_sections.shape[0] - 1)])
        p2 = np.array([self.patch_top_diffuser, 0])
        p3 = np.array([self.patch_top_diffuser, 0.2])

        t = np.linspace(np.zeros(2), np.ones(2), 100)
        t_bezier = ((1 - t) ** 2 * np.tile(p1, (100, 1)) +
                    2 * (1 - t) * t * np.tile(p3, (100, 1)) +
                    t ** 2 * np.tile(p2, (100, 1)))
        return self.parent.exit_diff.surface(t_bezier)

    @property
    def filler_top_curve2(self):
        p1 = self.parent.sections[self.start_angle][0][-1]
        p2 = self.parent.sections[self.start_angle][0][-1] + 10 * (
                self.parent.sections[self.start_angle - 1][0][-1] - self.parent.sections[self.start_angle][0][
            -1])
        p3 = self.parent.exit_diff.surface((self.patch_top_volute, 0.1))
        p4 = self.parent.exit_diff.surface((self.patch_top_volute, 0))
        t = np.linspace(np.zeros(3), np.ones(3), 100)
        return ((1 - t) ** 3 * np.tile(p1, (100, 1)) +
                3 * (1 - t) ** 2 * t * np.tile(p2, (100, 1)) +
                3 * (1 - t) * t ** 2 * np.tile(p3, (100, 1)) +
                t ** 3 * np.tile(p4, (100, 1)))

    @property
    def filler_top_curve3(self):
        angle = np.linspace(0, self.start_angle, 100)
        r = self.parent.sections[self.start_angle][0].r[0]
        x = r * np.cos(np.deg2rad(angle))
        y = r * np.sin(np.deg2rad(angle))
        z = np.full_like(y, self.parent.sections[self.start_angle][0].z[0])
        return np.vstack((x, y, z)).T

    @property
    def filler_bottom_curve1(self):
        t_revised = (1 - 1 / (len(self.parent.sections[0]) + 1)) - self.patch_bottom_diffuser
        index = np.array(
            np.where((self.parent.exit_diff.all_sections == tuple(self.edge_diffuser[-1])).all(axis=2)))[
                :, 0]
        p1 = np.array([index[1] / (self.parent.exit_diff.all_sections.shape[1] - 1),
                       index[0] / (self.parent.exit_diff.all_sections.shape[0] - 1)])
        p2 = np.array([t_revised, 0])
        p3 = np.array([t_revised, 0.2])

        t = np.linspace(np.zeros(2), np.ones(2), 100)
        t_bezier = ((1 - t) ** 2 * np.tile(p1, (100, 1)) +
                    2 * (1 - t) * t * np.tile(p3, (100, 1)) +
                    t ** 2 * np.tile(p2, (100, 1)))
        return self.parent.exit_diff.surface(t_bezier)

    @property
    def filler_bottom_curve2(self):
        t_revised = (1 - 1 / (len(self.parent.sections[0]) + 1)) - self.patch_bottom_volute
        p1 = self.parent.sections[self.start_angle][-1][0]
        p2 = self.parent.sections[self.start_angle][-1][0] + 10 * (
                self.parent.sections[self.start_angle - 1][-1][0] - self.parent.sections[self.start_angle][-1][
            0])
        p3 = self.parent.exit_diff.surface((t_revised, 0.1))
        p4 = self.parent.exit_diff.surface((t_revised, 0))
        t = np.linspace(np.zeros(3), np.ones(3), 100)
        return ((1 - t) ** 3 * np.tile(p1, (100, 1)) +
                3 * (1 - t) ** 2 * t * np.tile(p2, (100, 1)) +
                3 * (1 - t) * t ** 2 * np.tile(p3, (100, 1)) +
                t ** 3 * np.tile(p4, (100, 1)))

    @property
    def filler_bottom_curve3(self):
        angle = np.linspace(0, self.start_angle, 100)
        r = self.parent.sections[self.start_angle][-1].r[-1]
        x = r * np.cos(np.deg2rad(angle))
        y = r * np.sin(np.deg2rad(angle))
        z = np.full_like(y, self.parent.sections[self.start_angle][-1].z[-1])
        return np.vstack((x, y, z)).T

    def calc_radius(self):
        self.radius = ((1.03 + 0.1 * self.parent.impeller1D.specific_speed / 40 +
                        0.07 * self.parent.fluid.density / 1000 * self.parent.impeller1D.head_rise / 1000) *
                       self.parent.impeller1D.tip.outlet.radius)  # Gülich Table 10.2


class Diffuser3D:
    def __init__(self, diffuser1D: Diffuser):
        self.width = 0


class Pump3D:
    def __init__(self, pump1D: Pump):
        self.orientation = "exit-on-turbine"
        self.pump1D = pump1D
        self.components = []
        if self.pump1D.inducer_need:
            self.components.append(Inducer3D(self.pump1D.inducer))
            self.components[-1].initialize()
        self.components.append(Impeller3D(self.pump1D.impeller))
        self.components[-1].initialize()
        if self.pump1D.is_double_suction:
            self.impeller_disk_thickness = 2e-3
        if self.pump1D.is_second_stage:
            self.components.append(Diffuser3D(self.pump1D.diffuser))
            self.components.append(Impeller3D(self.pump1D.impeller2))
            self.components[-1].initialize()
        self.components.append(Volute3D(self.pump1D.volute, impeller3D=self.impeller))

    @property
    def inducer(self):
        return self.components[0]

    @property
    def impeller(self):
        if self.pump1D.inducer_need:
            return self.components[1]
        else:
            return self.components[0]

    @property
    def volute(self):
        return self.components[-1]

    @property
    def diffuser(self):
        if self.pump1D.is_second_stage:
            if self.pump1D.inducer_need:
                return self.components[2]
            else:
                return self.components[1]
        else:
            return None

    @property
    def impeller2(self):
        if self.pump1D.is_second_stage:
            if self.pump1D.inducer_need:
                return self.components[3]
            else:
                return self.components[2]
        else:
            return None

    @property
    def axial_locations(self):
        loc = []
        if self.pump1D.inducer_need:
            loc.append(np.array([0, 0, 0]))
        loc.append(np.array([0, 0, 0]))
        if self.pump1D.inducer_need:
            loc[-1] = np.array([0, 0, self.inducer.hub.z[-1]])
        loc.append(np.array([0, 0, loc[-1][2] + self.volute.anchor_z]))
        return loc

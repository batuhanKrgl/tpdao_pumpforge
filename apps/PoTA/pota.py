import numpy as np
import os
from pathlib import Path
import pickle
import json
import sys
from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item

from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QApplication, QMessageBox, QWidget, QHeaderView, QDoubleSpinBox, \
    QPushButton, QAbstractItemView, QMainWindow, QFileDialog, QLineEdit, QTableWidgetItem, QRadioButton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import image
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection, PathCollection

from src.Pump import pump1D
from src.Pump.pump3D import Pump3D
from src.Utility.obj_to_dict import NumpyEncoder, transform_to_dict
from src.fluid import Fluid
from src.Utility.plot_on_cordier import cordier_plot_on
from src.Utility.resource_call import resource_path
from src.Pump.pump1D import Pump


class PointCloudGeometry:
    # Pompa komponentlerinin 3D görüntü sınıfı
    def __init__(self, surfaces=None):
        self.opacity = 1
        self.surfaces = surfaces
        self.hidden = False
        self.offset = np.zeros(3)
        if surfaces is None:
            self.surfaces = []

    def set_opacity(self, opacity):
        for surf in self.surfaces:
            surf.opacity = opacity

    def set_color(self, color):
        for surf in self.surfaces:
            surf.color = color

    def set_offset(self, offset):
        self.offset = np.array(offset)
        for surface in self.surfaces:
            surface.set_offset(self.offset)

    def set_symetric(self, z_loc, angle):
        sym_surfaces = []
        for surface in self.surfaces:
            rotation_matrice = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]])
            symmetry_matrice = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]])
            # transform_matrice = rotation_matrice @ symmetry_matrice
            new_surface = []
            for curve in surface.curves:
                c1 = curve @ rotation_matrice.T
                c1[:, 2] += self.offset[2]
                c2 = c1 @ symmetry_matrice.T
                c2[:, 2] += 2 * z_loc
                new_surface.append(c2)
            sym_surfaces.append(PointCloudSurface(new_surface))
        return sym_surfaces

    def set_blades(self, blade_number):
        periodic_surfaces = []
        for i in range(blade_number):
            for surface in self.surfaces:
                angle = i / blade_number * 2 * np.pi
                rotation_matrice = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
                new_surface = []
                for curve in surface.curves:
                    new_surface.append(curve @ rotation_matrice.T)
                periodic_surfaces.append(PointCloudSurface(new_surface))
        self.surfaces = periodic_surfaces

    def revolve(self, curve: np.ndarray, num_rotations=100):
        # 100 adet döndürme açısı (0'dan 2π'ye kadar)
        angles = np.linspace(0, 2 * np.pi, num_rotations, endpoint=True)

        # Döndürme matrislerini oluşturalım
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        rotation_matrices = np.array([
            [cos_angles, -sin_angles, np.zeros_like(cos_angles)],
            [sin_angles, cos_angles, np.zeros_like(sin_angles)],
            [np.zeros_like(cos_angles), np.zeros_like(cos_angles), np.ones_like(cos_angles)]
        ]).transpose(2, 0, 1)  # (100, 3, 3) boyutunda

        # Eğrinin her noktasını her döndürme matrisiyle çarpmak için genişletelim
        expanded_curve = np.tile(curve, (num_rotations, 1, 1))  # (100, 100, 3) boyutunda

        # Döndürme işlemini gerçekleştirelim
        surf = np.einsum('ijk,ilk->ijl', expanded_curve, rotation_matrices)
        self.surfaces.append(PointCloudSurface([surf[i] for i in range(surf.shape[0])]))


class PointCloudLine:
    def __init__(self, points=None, opacity=1, color=(1, 0, 0)):
        self.line = None
        self.opacity = opacity
        self.color = color
        self.points = points
        self.first_time = True

    def set_opacity(self, opacity):
        self.line.actor.property.opacity = opacity

    def plot_line(self):
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        self.line = mlab.plot3d(x, y, z, opacity=self.opacity, color=self.color, tube_radius=0.0002)
        self.first_time = False

    def update_line(self):
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        self.line.mlab_source.set(x=x, y=y, z=z)


class PointCloudSurface:
    #  Komponentlerdeki yüzeylerin 3D görüntü sınıfı
    def __init__(self, curves=None, opacity=1, color=(1, 1, 1)):
        self.surface = None
        self.opacity = opacity
        self.color = color
        self.curves = curves
        self.offset = np.zeros(3)
        self.rotation = np.zeros(3)
        self.first_time = True
        if curves is None:
            self.curves = []

    def set_offset(self, offset):
        self.offset = offset

    def plot_surface(self):
        surface_points = np.concatenate(self.curves, axis=0)
        x, y, z = surface_points[:, 0], surface_points[:, 1], surface_points[:, 2]
        x += self.offset[0]
        y += self.offset[1]
        z += self.offset[2]
        self.surface = mlab.triangular_mesh(x, y, z, self.triangulate(), color=self.color)
        self.first_time = False

    def update_surface(self):
        surface_points = np.concatenate(self.curves, axis=0)
        x, y, z = surface_points[:, 0], surface_points[:, 1], surface_points[:, 2]
        x += self.offset[0]
        y += self.offset[1]
        z += self.offset[2]
        self.surface.actor.property.opacity = self.opacity
        self.surface.actor.property.color = self.color
        self.surface.mlab_source.set(x=x, y=y, z=z, triangles=self.triangulate())

    def triangulate(self):
        num_curves = len(self.curves)
        num_points = len(self.curves[0])
        triangles = []

        for i in range(num_curves - 1):
            for j in range(num_points - 1):
                triangles.append([i * num_points + j, (i + 1) * num_points + j, i * num_points + j + 1])
                triangles.append([(i + 1) * num_points + j, (i + 1) * num_points + j + 1, i * num_points + j + 1])

        return np.array(triangles)


class Visualization(HasTraits):
    # Mayavi Sahne Sınıfı
    scene = Instance(MlabSceneModel, ())
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), show_label=False),
                width=800, height=600, resizable=True)


def parse_float(text: str, field_name: str) -> float:
    cleaned_text = text.strip().replace(",", ".")
    try:
        return float(cleaned_text)
    except ValueError as exc:
        raise ValueError(f"Invalid value for field: {field_name}") from exc


def require_fields(dialog: QDialog, names: list) -> dict:
    missing = []
    values = {}
    for name in names:
        line_edit = dialog.findChild(QLineEdit, name)
        if line_edit is None or not line_edit.text().strip():
            missing.append(name)
            if line_edit is not None:
                line_edit.setStyleSheet("background-color: pink;")
            continue
        line_edit.setStyleSheet("")
        values[name] = line_edit.text()
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    return values


def build_pump_inputs_from_ui(dialog: QDialog):
    required_fields = [
        "lineEdit_revolution",
        "lineEdit_mass_flow",
        "lineEdit_outlet_pressure",
        "lineEdit_inlet_pressure",
        "lineEdit_inlet_angle",
        "lineEdit_inlet_diameter",
        "lineEdit_outlet_diameter",
        "lineEdit_density",
        "lineEdit_dynamicViscosity",
        "lineEdit_vaporPressure",
    ]
    values = require_fields(dialog, required_fields)

    rpm = parse_float(values["lineEdit_revolution"], "lineEdit_revolution")
    mass_flow = parse_float(values["lineEdit_mass_flow"], "lineEdit_mass_flow")
    outlet_pressure_bar = parse_float(values["lineEdit_outlet_pressure"], "lineEdit_outlet_pressure")
    inlet_pressure_bar = parse_float(values["lineEdit_inlet_pressure"], "lineEdit_inlet_pressure")
    inlet_angle = parse_float(values["lineEdit_inlet_angle"], "lineEdit_inlet_angle")
    inlet_diameter_mm = parse_float(values["lineEdit_inlet_diameter"], "lineEdit_inlet_diameter")
    outlet_diameter_mm = parse_float(values["lineEdit_outlet_diameter"], "lineEdit_outlet_diameter")
    density = parse_float(values["lineEdit_density"], "lineEdit_density")
    dynamic_viscosity = parse_float(values["lineEdit_dynamicViscosity"], "lineEdit_dynamicViscosity")
    vapor_pressure = parse_float(values["lineEdit_vaporPressure"], "lineEdit_vaporPressure")

    pump_dict = {
        "mass_flow": mass_flow,
        "pressure_required": outlet_pressure_bar * 100000,
        "inlet_pressure": inlet_pressure_bar * 100000,
        "alpha": inlet_angle,
        "double_suction": dialog.checkBox_double_suction.isChecked(),
        "second_stage": dialog.checkBox_second_stage.isChecked(),
        "inlet_radius": inlet_diameter_mm / 2000,
        "outlet_radius": outlet_diameter_mm / 2000,
        "shaft_radius": None,
    }
    fluid_props = {
        "density": density,
        "dynamicViscosity": dynamic_viscosity,
        "vaporPressure": vapor_pressure,
    }
    return rpm, pump_dict, fluid_props


class MayaviQWidget(QWidget):
    # Mayavi sahnesi entegre edilmiş QWidget
    def __init__(self, parent=None):
        super(MayaviQWidget, self).__init__(parent)
        self.visualization = Visualization()
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.visualization.edit_traits(parent=self, kind='subpanel').control)
        self.geometries = []
        self.lines = []

    def add_geometry(self, geometry):
        self.geometries.append(geometry)

    def add_line(self, line):
        self.lines.append(line)

    def update_viewer(self):
        for geo in self.geometries:
            if len(geo.surfaces) > 0:
                for surface in geo.surfaces:
                    if surface.first_time:
                        surface.plot_surface()
                    else:
                        surface.update_surface()
        for line in self.lines:
            if line.first_time:
                line.plot_line()
            else:
                line.update_line()


class Polygon:
    # Layout gösterimi için komponentlerin veri sınıfı
    def __init__(self, points=np.array([])):
        self.points = points

    def update_points(self, new_points):
        self.points = new_points


class GeometryPlot(QWidget):
    def __init__(self, parent=None, got_axline=True, rz=True):
        super(GeometryPlot, self).__init__(parent)
        self.figure = Figure()
        # self.figure.tight_layout(pad=1, rect=[-0.5,-0.5,1.5,1.5])
        self.figure.patch.set_alpha(0)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background:transparent;")
        self.canvas.setAttribute(Qt.WA_TranslucentBackground, True)
        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111, position=[0.0, 0.01, 0.999, 0.99])
        if rz:
            self.figure.text(0.05, 0.97, "r [mm]", rotation=0, fontweight="bold")
            self.figure.text(0.895, 0.1, "z [mm]", fontweight="bold")
        else:
            self.figure.text(0.05, 0.97, "y [mm]", rotation=0, fontweight="bold")
            self.figure.text(0.895, 0.1, "x [mm]", fontweight="bold")

        self.ax.tick_params(axis="y", direction="in", pad=-25)
        self.ax.tick_params(axis="x", direction="in", pad=-15)
        if got_axline:
            self.ax.axvline(x=0, color="black", linestyle="--", linewidth="0.5")
            self.ax.axhline(y=0, color="black", linestyle="--", linewidth="0.5")
            # self.ax.axline((0, 0), (1, 0), linestyle="--")
            # self.ax.axline((0, 0), (0, 1), linestyle="--")
        self.ax.grid(linestyle='--', linewidth=0.5)
        self.ax.set_aspect('equal', "datalim")
        self.lines = []

    def update_plot(self):
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.canvas.draw()


class LayoutPlot(QWidget):
    # Matplotlib gömülmüş ve layout gösterimi için özelleştirilmiş QWidget
    def __init__(self, parent=None):
        super(LayoutPlot, self).__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(linestyle='--', linewidth=0.5)
        self.ax.axline((0, 0), (1, 0), linestyle="--")
        self.ax.set_aspect('equal', "datalim")
        self.ax.set_xlim([0, 2])
        # self.ax_layout.add_patch(self.impeller_poly)
        # self.ax_layout.axes.get_xaxis().set_visible(False)
        # self.ax_layout.axes.get_yaxis().set_visible(False)
        self.figure.tight_layout()
        self.polygons = []

    def add_polygon(self, polygon):
        self.polygons.append(polygon)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        for polygon in self.polygons:
            if polygon.points.size > 0:
                self.ax.fill(polygon.points[:, 0], polygon.points[:, 1], alpha=0.5)
        self.ax.grid(linestyle='--', linewidth=0.5)
        self.ax.axline((0, 0), (1, 0), linestyle="--")
        self.ax.set_aspect('equal', "datalim")
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()


class PotaMain(QMainWindow):
    # Pompa Tasarım Aracı Ana Pencere
    closed = pyqtSignal()

    def __init__(self):
        super(PotaMain, self).__init__()
        loadUi(resource_path('resource/PoTA-Main.ui'), self)
        self.showMaximized()

        self.layout_plot = LayoutPlot(self)
        self.layout_layout.addWidget(NavigationToolbar(self.layout_plot.canvas, self))
        self.layout_layout.addWidget(self.layout_plot.canvas)

        self.viewer_3D = MayaviQWidget(self)
        self.tab_3d_layout.addWidget(self.viewer_3D)

        self.pushButton_create.clicked.connect(self.exec_setup_dialog)
        self.pushButton_inducer.clicked.connect(self.design_inducer_3d)
        self.pushButton_impeller.clicked.connect(self.design_impeller_3d)
        self.pushButton_diffuser.clicked.connect(self.exec_design_dialog)
        self.pushButton_impeller2.clicked.connect(self.exec_design_dialog)
        self.pushButton_volute.clicked.connect(self.design_volute_3d)

        self.pushButton_inducer.hide()
        self.pushButton_impeller.hide()
        self.pushButton_diffuser.hide()
        self.pushButton_impeller2.hide()
        self.pushButton_volute.hide()

        self.action_save_object.triggered.connect(self.save_object)
        self.action_save_json.triggered.connect(self.save_json)
        self.action_open_object.triggered.connect(self.open_object)
        self.action_open_json.triggered.connect(self.open_json)

        self.write_path = None
        self.pump3D = None
        self.pump = None

        self.impeller_poly = Polygon()
        self.inducer_poly = Polygon()
        self.volute_poly = Polygon()
        self.impeller2_poly = Polygon()
        self.impeller_s_poly = Polygon()
        self.inducer_s_poly = Polygon()

        self.impeller_geo = PointCloudGeometry()
        self.inducer_geo = PointCloudGeometry()
        self.volute_geo = PointCloudGeometry()
        self.impeller2_geo = PointCloudGeometry()
        self.impeller_s_geo = PointCloudGeometry()
        self.inducer_s_geo = PointCloudGeometry()

        self.layout_plot.add_polygon(self.impeller_poly)
        self.layout_plot.add_polygon(self.inducer_poly)
        self.layout_plot.add_polygon(self.volute_poly)
        self.layout_plot.add_polygon(self.impeller2_poly)
        self.layout_plot.add_polygon(self.impeller_s_poly)
        self.layout_plot.add_polygon(self.inducer_s_poly)

        self.viewer_3D.add_geometry(self.impeller_geo)
        self.viewer_3D.add_geometry(self.inducer_geo)
        self.viewer_3D.add_geometry(self.volute_geo)
        self.viewer_3D.add_geometry(self.impeller2_geo)
        self.viewer_3D.add_geometry(self.impeller_s_geo)
        self.viewer_3D.add_geometry(self.inducer_s_geo)

    def design_impeller_3d(self):
        dialog = ImpellerDialog(resource_path('resource/PoTA-Impeller.ui'), self.pump.impeller, None, self)
        dialog.exec_()
        self.draw_pump()

    def design_inducer_3d(self):
        dialog = ImpellerDialog(resource_path('resource/inducer.ui'), None, self.pump.inducer, self)
        dialog.exec_()
        self.draw_pump()

    def design_volute_3d(self):
        dialog = VoluteDialog(resource_path('resource/PoTA-Volute.ui'), self, self.pump3D)
        dialog.exec_()
        self.draw_pump()

    def exec_design_dialog(self):
        QMessageBox.information(self, "Message", f"İçerik Hazırlanıyor.")

    def exec_setup_dialog(self):
        dialog = PotaSetupDialog()
        if dialog.exec_() == QDialog.Accepted:
            self.pump = dialog.pump
            if self.pump.inducer_need:
                self.pushButton_inducer.show()
            self.pushButton_impeller.show()
            if self.pump.is_second_stage:
                self.pushButton_diffuser.show()
                self.pushButton_impeller2.show()
            self.pushButton_volute.show()
            self.pump3D = Pump3D(self.pump)
            self.draw_pump()

    def draw_pump(self):
        impeller_x_points = np.concatenate((self.pump3D.impeller.hub.z,
                                            self.pump3D.impeller.tip.z[::-1])) * 1000
        impeller_x_points += self.pump3D.axial_locations[1][-1] * 1000
        impeller_y_points = np.concatenate((self.pump3D.impeller.hub.r,
                                            self.pump3D.impeller.tip.r[::-1])) * 1000
        self.impeller_poly.update_points(np.array([impeller_x_points, impeller_y_points]).transpose())

        volute_points_x = np.concatenate(tuple([curve.z * 1000 for curve in self.pump3D.volute.sections[-1]]))
        volute_points_x = np.concatenate((volute_points_x, [volute_points_x[0]]))
        volute_points_y = np.concatenate(tuple([curve.r * 1000 for curve in self.pump3D.volute.sections[-1]]))
        volute_points_y = np.concatenate((volute_points_y, [volute_points_y[0]]))
        self.volute_poly.update_points(
            np.array([volute_points_x + self.pump3D.axial_locations[-1][-1] * 1000, volute_points_y]).T)

        inducer_x_points = np.concatenate((self.pump3D.inducer.hub.z,
                                           self.pump3D.inducer.tip.z[::-1])) * 1000
        inducer_x_points += self.pump3D.axial_locations[0][-1] * 1000
        inducer_y_points = np.concatenate((self.pump3D.inducer.hub.r,
                                           self.pump3D.inducer.tip.r[::-1])) * 1000
        self.inducer_poly.update_points(np.array([inducer_x_points, inducer_y_points]).transpose())
        self.viewer_3D.visualization.scene.mlab.clf()
        self.impeller_geo.surfaces = [
            PointCloudSurface(self.pump3D.impeller.foil_dict["pressure"]["curves"]),
            PointCloudSurface(self.pump3D.impeller.foil_dict["suction"]["curves"]),
            PointCloudSurface(self.pump3D.impeller.leading_edge_dict["curves"]),
            PointCloudSurface(self.pump3D.impeller.trailing_edge_dict["curves"])]
        self.impeller_geo.set_blades(6)
        self.impeller_geo.revolve(self.pump3D.impeller.hub)
        self.impeller_geo.revolve(self.pump3D.impeller.tip)
        self.impeller_geo.set_offset([0, 0, self.pump3D.axial_locations[1][-1]])

        surfaces = []
        for i in range(len(self.pump3D.volute.sections[0])):
            surf = []
            for sec in self.pump3D.volute.sections[self.pump3D.volute.cut_water.start_angle:]:
                surf.append(sec[i])
            surfaces.append(PointCloudSurface(surf))
        surfaces.append(PointCloudSurface(self.pump3D.volute.exit_diff.all_sections.tolist()))
        surfaces.append(PointCloudSurface(self.pump3D.volute.cut_water.surface.tolist()))
        self.volute_geo.surfaces = surfaces
        self.volute_geo.set_offset([0, 0, self.pump3D.axial_locations[-1][-1]])

        self.inducer_geo.surfaces = [
            PointCloudSurface(self.pump3D.inducer.foil_dict["pressure"]["curves"]),
            PointCloudSurface(self.pump3D.inducer.foil_dict["suction"]["curves"]),
            PointCloudSurface(self.pump3D.inducer.leading_edge_dict["curves"]),
            PointCloudSurface(self.pump3D.inducer.trailing_edge_dict["curves"])]
        self.inducer_geo.set_blades(3)
        self.inducer_geo.revolve(self.pump3D.inducer.hub)
        self.inducer_geo.revolve(self.pump3D.inducer.tip)
        self.inducer_geo.set_offset([0, 0, self.pump3D.axial_locations[0][-1]])

        if self.pump.is_double_suction:
            impeller_s_x_points = impeller_x_points + 2 * (np.max(impeller_x_points) + 1 - impeller_x_points)
            inducer_s_x_points = inducer_x_points + 2 * (np.max(impeller_x_points) + 1 - inducer_x_points)
            self.impeller_s_poly.update_points(np.array([impeller_s_x_points, impeller_y_points]).transpose())
            self.inducer_s_poly.update_points(np.array([inducer_s_x_points, inducer_y_points]).transpose())

            symmetry_z = (self.pump3D.impeller.hub.z[-1] + self.pump3D.inducer.hub.z[-1] +
                          self.pump3D.impeller_disk_thickness / 2)
            self.impeller_s_geo.surfaces = self.impeller_geo.set_symetric(z_loc=symmetry_z, angle=2 * np.pi / 12)
            self.inducer_s_geo.surfaces = self.inducer_geo.set_symetric(z_loc=symmetry_z, angle=2 * np.pi / 12)

        self.layout_plot.update_plot()
        self.viewer_3D.update_viewer()

    def save_object(self):
        desktop_path = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Object", os.path.join(desktop_path, "test.pump3D"),
                                                   "Pump3D Files (*.pump3D);;All Files (*)", options=options)
        if file_path:
            with open(file_path, "wb") as file:
                pickle.dump(self.pump3D, file)

    def open_object(self):
        desktop_path = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Object", desktop_path,
                                                   "Pump3D Files (*.pump3D);;All Files (*)", options=options)
        if file_path:
            with open(file_path, "rb") as file:
                self.pump3D = pickle.load(file)
            self.draw_pump()

    def save_json(self):
        desktop_path = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save JSON File", os.path.join(desktop_path, "test.json"),
                                                   "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            with open(file_path, "w") as file:
                dict = transform_to_dict(self.pump3D)
                json.dump(dict, file, indent=5, cls=NumpyEncoder)

    def open_json(self):
        desktop_path = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Open JSON File", desktop_path,
                                                   "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            with open(file_path, "r") as file:
                dict = transform_to_dict(self.pump3D)
                json.dump(dict, file, indent=5, cls=NumpyEncoder)

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()


class PotaSetupDialog(QDialog):
    # Pompa 1D obje oluşturma veya okutma için QDialog
    def __init__(self):
        super(PotaSetupDialog, self).__init__()

        self.inputs_changed = False
        self.pump = None
        self.fluid = Fluid()
        fluid_files = os.listdir(resource_path("resource/fluids/oxidizer/"))
        fluid_files = fluid_files + os.listdir(resource_path("resource/fluids/fuel/"))

        loadUi(resource_path('resource/PoTA-Setup.ui'), self)
        self.comboBox.clear()
        self.comboBox.addItem("Kullanıcı Tanımlı")
        self.comboBox.addItems([os.path.splitext(f)[0] for f in fluid_files])
        self.comboBox.currentTextChanged.connect(self.handle_combo_box_change)

        for line_edit in self.findChildren(QLineEdit):
            line_edit.textChanged.connect(self.on_input_changed)

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tableWidget.update()

        self.figure_cordier = Figure()
        self.figure_cordier.patch.set_alpha(0)
        self.canvas_cordier = FigureCanvas(self.figure_cordier)
        self.canvas_cordier.setStyleSheet("background:transparent;")
        self.canvas_cordier.setAttribute(Qt.WA_TranslucentBackground, True)
        self.ax_cordier = self.figure_cordier.add_subplot(111)
        self.ax_cordier.axes.get_xaxis().set_visible(False)
        self.ax_cordier.axes.get_yaxis().set_visible(False)
        self.figure_cordier.tight_layout()
        self.ax_cordier.set_ylim([1181, 0])
        self.ax_cordier.set_xlim([0, 1776])
        self.ax_cordier.imshow(image.imread(resource_path("resource/cordier_small.jpg")))
        self.groupBox.layout().replaceWidget(self.widget_dummy, self.canvas_cordier)

        self.pushButton_import.clicked.connect(self.import_pump)
        self.pushButton_calculate.clicked.connect(self.calc_pump)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def import_pump(self):
        # import bypass için aşağıdaki iki satırı yorumdan çıkar/ yoruma ekle.
        file_path = "C:\\Users\\batuhan.koroglu\\Desktop\\TPDAO\\apps\\TuKTA\\test-save\\250kN_70bar-20240813-073438\\250kN_70bar-20240813-073438-21000rpm-oxidizer.pump"
        # file_path, _ = QFileDialog.getOpenFileName(parent=self, caption="Select Pump File", directory="./",
        #                                            filter="Pump 1D File (*.pump)")
        if file_path:
            with open(file_path, "rb") as file:
                pump = pickle.load(file)

            self.lineEdit_revolution.setText(f"{pump.rpm:.0f}")
            self.lineEdit_mass_flow.setText(f"{pump.mass_flow:.2f}")
            self.lineEdit_outlet_pressure.setText(f"{pump.station_3.total_pressure / 100000:.2f}")
            self.lineEdit_inlet_pressure.setText(f"{pump.station_0.total_pressure / 100000:.2f}")
            self.lineEdit_inlet_angle.setText(f"{pump.inlet_angle:.0f}")
            self.lineEdit_inlet_diameter.setText(f"{pump.inlet_radius * 2000:.2f}")
            self.lineEdit_outlet_diameter.setText(f"{pump.outlet_radius * 2000:.2f}")

            self.checkBox_second_stage.setChecked(pump.is_second_stage)
            self.checkBox_double_suction.setChecked(pump.is_double_suction)
            self.comboBox.setCurrentText(pump.fluid.name)
            self.handle_combo_box_change(pump.fluid.name)

            for line_edit in [self.lineEdit_density, self.lineEdit_dynamicViscosity, self.lineEdit_vaporPressure]:
                line_edit.setText(str(getattr(pump.fluid, line_edit.objectName().split("_")[-1])))

            self.pump = pump
            self.update_table()
            self.plot_cordier()
            self.inputs_changed = False

    def calc_pump(self):
        try:
            rpm, pump_dict, fluid_props = build_pump_inputs_from_ui(self)
        except ValueError as exc:
            QMessageBox.warning(self, "Geçersiz Giriş", str(exc))
            return False

        self.fluid.update_properties("density", fluid_props["density"])
        self.fluid.update_properties("vaporPressure", fluid_props["vaporPressure"])
        self.fluid.update_properties("dynamicViscosity", fluid_props["dynamicViscosity"])
        self.pump = Pump(rpm=rpm, pump_dict=pump_dict, fluid=self.fluid)
        self.update_table()
        self.plot_cordier()
        self.inputs_changed = False
        return True

    def handle_combo_box_change(self, fluid_name):
        if fluid_name in [os.path.splitext(f)[0] for f in os.listdir(resource_path("resource/fluids/oxidizer/"))]:
            fluid_file = resource_path(f"{'resource/fluids/oxidizer/'}/{fluid_name}.fluid")
        else:
            fluid_file = resource_path(f"{'resource/fluids/fuel/'}/{fluid_name}.fluid")

        if os.path.isfile(fluid_file):
            self.fluid.load_pump_properties(fluid_file)
            for line_edit in [self.lineEdit_density, self.lineEdit_dynamicViscosity, self.lineEdit_vaporPressure]:
                line_edit.setEnabled(False)
                line_edit.setStyleSheet("")
                line_edit.setText(str(getattr(self.fluid, line_edit.objectName().split("_")[-1])))
        else:
            for line_edit in [self.lineEdit_density, self.lineEdit_dynamicViscosity, self.lineEdit_vaporPressure]:
                line_edit.setEnabled(True)
                line_edit.setStyleSheet("")

        self.inputs_changed = True

    def on_input_changed(self):
        self.inputs_changed = True

    def accept(self):
        if self.inputs_changed or self.pump is None:
            if self.calc_pump():
                super().accept()
        else:
            super().accept()

    def reject(self):
        self.pump = None
        super().reject()

    def update_table(self):
        self.tableWidget.setItem(1, 2, QTableWidgetItem(f"{self.pump.rpm:.0f}"))
        self.tableWidget.setItem(2, 2, QTableWidgetItem(f"{self.pump.mass_flow / self.pump.fluid.density:.3f}"))
        self.tableWidget.setItem(3, 2, QTableWidgetItem(f"{self.pump.mass_flow:.2f}"))
        self.tableWidget.setItem(4, 2, QTableWidgetItem(
            f"{self.pump.station_3.total_pressure / self.pump.fluid.density / 9.805:.0f}"))
        self.tableWidget.setItem(5, 2, QTableWidgetItem(f"{self.pump.station_3.total_pressure / 100000:.2f}"))
        self.tableWidget.setItem(6, 2, QTableWidgetItem(f"{self.pump.specific_speed:.2f}"))
        self.tableWidget.setItem(7, 2, QTableWidgetItem(f"{self.pump.shaft_power / 1000:.2f}"))
        self.tableWidget.setItem(8, 2, QTableWidgetItem(f"{self.pump.hydraulic_efficiency:.2f}"))
        self.tableWidget.setItem(9, 2, QTableWidgetItem(f"{self.pump.inlet_radius * 2000:.0f}"))
        self.tableWidget.setItem(10, 2, QTableWidgetItem(f"{self.pump.outlet_radius * 2000:.0f}"))
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tableWidget.update()

    def plot_cordier(self):
        cordier_plot_on(self.ax_cordier, self.pump.impeller.specific_speed, self.pump.impeller.specific_diameter,
                        size="small")


class ImpellerDialog(QDialog):
    def __init__(self, ui_file, imp1D: pump1D.Impeller = None, ind1D: pump1D.Inducer = None, parent=None):
        super(ImpellerDialog, self).__init__()
        loadUi(ui_file, self)
        self.parent = parent
        self.sender_name = self.parent.sender().objectName()
        if ind1D is None:
            self.rotor1D = imp1D
            self.rotor3D = self.parent.pump3D.impeller

        else:
            self.rotor1D = ind1D
            self.rotor3D = self.parent.pump3D.inducer

        self.rotor3D.initialize()

        ########## WİDGET KURULUMU ##########

        for lineEdit in self.page_meridional.findChildren(QLineEdit, options=Qt.FindDirectChildrenOnly):
            lineEdit.setValidator(QDoubleValidator())
            key_value = self.rotor3D.meridional_dict[lineEdit.objectName().replace("lineEdit_", "")]
            lineEdit.setText(f"{key_value * 1000:.1f}")
            lineEdit.textChanged.connect(self.update_meridonal_dict_CP)

        self.lineEdit_hub_thickness.setText(f"{self.rotor3D.thickness_dict['hub']['thickness'][0] * 1000:.1f}")
        self.lineEdit_tip_thickness.setText(f"{self.rotor3D.thickness_dict['tip']['thickness'][0] * 1000:.1f}")

        self.lineEdit_hub_thickness.textChanged.connect(self.update_thickness_dict)
        self.lineEdit_tip_thickness.textChanged.connect(self.update_thickness_dict)

        self.doubleSpinBox_LE_guide_which.setMaximum(self.rotor3D.guides_dict["number_of_guides"])
        self.doubleSpinBox_hub_node.setMaximum(self.doubleSpinBox_hub_piece.value() + 1)
        self.doubleSpinBox_tip_node.setMaximum(self.doubleSpinBox_tip_piece.value() + 1)
        self.doubleSpinBox_hub_loc.setEnabled(False)
        self.doubleSpinBox_tip_loc.setEnabled(False)
        self.doubleSpinBox_leading_ratio.valueChanged.connect(self.update_leading_edge_dict)
        if self.rotor3D.type == "inducer":
            self.doubleSpinBox_trailing_ratio.valueChanged.connect(self.update_leading_edge_dict)

        self.doubleSpinBox_hub_inlet_CP: QDoubleSpinBox
        self.doubleSpinBox_hub_inlet_CP.setValue(self.rotor3D.meridional_dict["hub_inlet_CP"])
        self.doubleSpinBox_tip_inlet_CP.setValue(self.rotor3D.meridional_dict["tip_inlet_CP"])
        self.doubleSpinBox_hub_outlet_CP.setValue(self.rotor3D.meridional_dict["hub_outlet_CP"])
        self.doubleSpinBox_tip_outlet_CP.setValue(self.rotor3D.meridional_dict["tip_outlet_CP"])
        self.doubleSpinBox_LE_hub.setValue(self.rotor3D.meridional_dict["leading_edge_@hub"])
        self.doubleSpinBox_LE_tip.setValue(self.rotor3D.meridional_dict["leading_edge_@tip"])
        self.doubleSpinBox_LE_guide_which.setValue(self.rotor3D.meridional_dict["leading_edge_@guide"][0])
        self.doubleSpinBox_LE_guide.setValue(self.rotor3D.meridional_dict["leading_edge_@guide"][1])
        self.doubleSpinBox_TE_hub.setValue(self.rotor3D.meridional_dict["trailing_edge_@hub"])
        self.doubleSpinBox_TE_tip.setValue(self.rotor3D.meridional_dict["trailing_edge_@tip"])

        for spin_box in self.findChildren(QDoubleSpinBox):
            if spin_box.parent().objectName().split("_")[-1] == "bezierNodes":
                spin_box.valueChanged.connect(self.update_meridonal_dict_CP)
            elif spin_box.parent().objectName().split("_")[-1] == "betaCPs":
                spin_box.valueChanged.connect(self.update_beta_dict_CP)
            elif spin_box.parent().objectName().split("_")[-1] == "leadingMer":
                spin_box.valueChanged.connect(self.update_leading_edge)
            elif spin_box.parent().objectName().split("_")[-1] == "trailingMer":
                spin_box.valueChanged.connect(self.update_leading_edge)
            elif spin_box.parent().parent().objectName().split("_")[-1] == "thickness":
                spin_box.valueChanged.connect(self.update_thickness_dict)

        for push_button in self.findChildren(QPushButton):
            if push_button.parent().objectName().split("_")[-1] == "1D":
                push_button.clicked.connect(self.recall_1D_value)

        self.radioButton_inlet_lineer.toggled.connect(self.update_table_view)
        self.radioButton_inlet_free.toggled.connect(self.update_table_view)
        self.radioButton_outlet_lineer.toggled.connect(self.update_table_view)
        self.radioButton_outlet_free.toggled.connect(self.update_table_view)

        # self.pushButton_next.clicked.connect(self.switching_widgets)
        # self.pushButton_before.clicked.connect(self.switching_widgets)
        # self.pushButton_plot_blade.clicked.connect(self.plot_blade)
        # self.pushButton_export.clicked.connect(self.export_step)
        # self.pushButton_json.clicked.connect(self.export_dict)
        # self.pushButton_save.clicked.connect(self.export_object)

        self.comboBox_leading.currentIndexChanged.connect(self.handle_comboBox_change)
        self.comboBox_trailing.currentIndexChanged.connect(self.handle_comboBox_change)

        # self.stackedWidget.setCurrentIndex(0)
        self.stackedWidget_leading.setCurrentIndex(0)
        self.stackedWidget_trailing.setCurrentIndex(0)

        self.tableWidget_beta.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.tableWidget_beta.setRowCount(self.rotor3D.beta_dict["array"].shape[0])
        self.tableWidget_beta.horizontalHeader().setStretchLastSection(True)
        self.tableWidget_beta.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i in range(self.rotor3D.beta_dict["array"].shape[0]):
            self.tableWidget_beta.setItem(i, 0,
                                          QTableWidgetItem(
                                              f'{self.rotor3D.beta_dict["array"][i, 0] * 180 / np.pi:.2f}'))
            self.tableWidget_beta.setItem(i, 1,
                                          QTableWidgetItem(
                                              f'{self.rotor3D.beta_dict["array"][i, -1] * 180 / np.pi:.2f}'))
        self.update_table_view()

        self.viewer_meridional = GeometryPlot(self, got_axline=False)
        self.gridLayout_upper_1.addWidget(self.viewer_inlet)

        self.viewer_beta = GeometryPlot(self, rz=False)
        self.gridLayout_lower_1.addWidget(self.viewer_beta)

        self.viewer_3D = MayaviQWidget(self)
        self.gridLayout_3d.addWidget(self.viewer_3D)

        self.viewer_thickness = GeometryPlot(self, rz=False)
        self.gridLayout_upper_2.addWidget(self.viewer_thickness)

        self.viewer_leading_edges = GeometryPlot(self, rz=False)
        self.gridLayout_lower_2.addWidget(self.viewer_leading_edges)

        self.viewer_trailing_edges = GeometryPlot(self, rz=False)
        self.gridLayout_upper_3.addWidget(self.viewer_trailing_edges)

        self.hub_line = Line2D(xdata=[], ydata=[], color='blue', linestyle="-", label='Hub')
        self.hub_node_line = Line2D(xdata=[], ydata=[], color='blue', linestyle="-o", linewidth=0.5, ms=3)
        self.tip_line = Line2D(xdata=[], ydata=[], color='green', linestyle="-", label='Tip')
        self.tip_node_line = Line2D(xdata=[], ydata=[], color='green', linestyle="-o", linewidth=0.5, ms=3)
        self.leading_line = Line2D(xdata=[], ydata=[], color='red', linestyle="-", label='Leading Edge')
        self.leading_node_line = Line2D(xdata=[], ydata=[], color='red', linestyle="-", linewidth=0.5, ms=3)
        self.trailing_line = Line2D(xdata=[], ydata=[], color='black', linestyle="-", label='Trailing Edge')

        self.guide_lines = []
        for i in self.rotor3D.guides_dict["guides"]:
            self.guide_lines.append(Line2D(xdata=[], ydata=[], color='black', linestyle="--", linewidth=0.5))
            self.viewer_meridional.ax.add_line(self.guide_lines[-1])

        self.viewer_meridional.ax.add_line(self.hub_line)
        self.viewer_meridional.ax.add_line(self.hub_node_line)
        self.viewer_meridional.ax.add_line(self.tip_line)
        self.viewer_meridional.ax.add_line(self.tip_node_line)
        self.viewer_meridional.ax.add_line(self.leading_line)
        self.viewer_meridional.ax.add_line(self.leading_node_line)
        self.viewer_meridional.ax.add_line(self.trailing_line)

        self.beta_inlet_nodes = Line2D(xdata=[], ydata=[], color='green', linestyle="-o", linewidth=0.5, ms=3)
        self.viewer_beta.ax.add_line(self.beta_inlet_nodes)

        self.beta_lines = []
        for i, beta_line in enumerate(self.rotor3D.beta_dict["array"]):
            color = 'black'
            linestyle = "--"
            if i in [0]:
                color = 'green'
                linestyle = "-"
            if i in [self.rotor3D.beta_dict["array"].shape[0] - 1]:
                color = 'blue'
                linestyle = "-"
            self.beta_lines.append(Line2D(xdata=[], ydata=[], color=color, linestyle=linestyle))
            self.viewer_beta.ax.add_line(self.beta_lines[-1])

        self.hub_merid = PointCloudLine()
        self.tip_merid = PointCloudLine()
        self.blade_hub = PointCloudLine()
        self.blade_tip = PointCloudLine()

        self.hub_merid.points = self.rotor3D.hub
        self.tip_merid.points = self.rotor3D.tip
        self.blade_hub.points = self.rotor3D.blade_hub
        self.blade_tip.points = self.rotor3D.blade_tip

        self.viewer_3D.ax.add_line(self.hub_merid)
        self.viewer_3D.ax.add_line(self.tip_merid)
        self.viewer_3D.ax.add_line(self.blade_hub)
        self.viewer_3D.ax.add_line(self.blade_tip)

        self.blade_lines = []
        for i, blade_line in enumerate(self.rotor3D.guides_dict["blade_guides"]):
            self.blade_lines.append(PointCloudLine(blade_line))
            self.viewer_3D.ax.add_line(self.blade_lines[-1])

        circle = np.linspace(0, 2 * np.pi)
        self.impeller_hub_exit = PointCloudLine(np.concatenate((self.rotor3D.hub.r[-1] * np.cos(circle),
                                                                self.rotor3D.hub.r[-1] * np.sin(circle),
                                                                self.rotor3D.hub.z[-1])))
        self.impeller_hub_inlet = PointCloudLine(np.concatenate((self.rotor3D.hub.r[0] * np.cos(circle),
                                                                 self.rotor3D.hub.r[0] * np.sin(circle),
                                                                 self.rotor3D.hub.z[0])))
        self.impeller_tip_exit = PointCloudLine(np.concatenate((self.rotor3D.tip.r[-1] * np.cos(circle),
                                                                self.rotor3D.tip.r[-1] * np.sin(circle),
                                                                self.rotor3D.tip.z[-1])))
        self.impeller_tip_inlet = PointCloudLine(np.concatenate((self.rotor3D.tip.r[0] * np.cos(circle),
                                                                 self.rotor3D.tip.r[0] * np.sin(circle),
                                                                 self.rotor3D.tip.z[0])))
        self.viewer_3D.ax.add_line(self.impeller_hub_exit)
        self.viewer_3D.ax.add_line(self.impeller_hub_inlet)
        self.viewer_3D.ax.add_line(self.impeller_tip_exit)
        self.viewer_3D.ax.add_line(self.impeller_tip_inlet)

        self.pressure_lines = []
        self.suction_lines = []
        for pressure_line, suction_line in zip(self.rotor3D.foil_dict["pressure"]["curves"],
                                               self.rotor3D.foil_dict["suction"]["curves"]):

            self.pressure_lines.append(PointCloudLine(pressure_line))
            self.suction_lines.append(PointCloudLine(suction_line))
            self.viewer_3D.ax.add_line(self.pressure_lines[-1])
            self.viewer_3D.ax.add_line(self.suction_lines)

        self.leading_lines = []
        for i, edge_line in enumerate(self.rotor3D.leading_edge_dict["curves"]):
            self.edge_line.append(PointCloudLine(edge_line))
            self.viewer_3D.ax.add_line(self.edge_line[-1])

        self.trailing_lines = []
        if self.rotor3D.trailing_edge_dict["method"] == "Eliptik":
            self.comboBox_trailing.setCurrentIndex(1)
            for i, edge_line in enumerate(self.rotor3D.trailing_edge_dict["curves"]):
                self.edge_line.append(PointCloudLine(edge_line))
                self.viewer_3D.ax.add_line(self.edge_line[-1])

        viewer_thickness

        self.hub_pressure_line = Line2D(xdata=[], ydata=[], color='blue', linestyle="-o", ms=2)
        self.hub_suction_line = Line2D(xdata=[], ydata=[], color='blue', linestyle="-o", ms=2)
        self.tip_pressure_line = Line2D(xdata=[], ydata=[], color='green', linestyle="-o", ms=2)
        self.tip_suction_line = Line2D(xdata=[], ydata=[], color='green', linestyle="-o", ms=2)


        self.hub_hatch = PolyCollection(verts=[], interpolate=True, color=colors.CSS4_COLORS["lightblue"], hatch="//",
                                        edgecolor="red")

        self.tip_hatch = PolyCollection(verts=[], interpolate=True, color=colors.CSS4_COLORS["lightgreen"], hatch="\\",
                                        edgecolor="green")

        self.hub_pointer = PathCollection(paths=[], s=100, c='blue')
        self.tip_pointer = PathCollection(paths=[], s=100, c='green')
        t = np.linspace(0.5 * np.pi, 1.5 * np.pi, 20)
        [self.leading_edge] = self.canvas_edges.ax1.plot(self.rotor3D.leading_edge_dict["ratio"] * np.cos(t), np.sin(t),
                                                         color="red")
        [self.leading_edge_p] = self.canvas_edges.ax1.plot([0, 1], [1, 1], color="black", linestyle="--")
        [self.leading_edge_s] = self.canvas_edges.ax1.plot([0, 1], [-1, -1], color="black", linestyle="--")
        if self.rotor3D.trailing_edge_dict["method"] == "Eliptik":
            [self.trailing_edge] = self.canvas_edges.ax2.plot(1 - self.rotor3D.trailing_edge_dict["ratio"] * np.cos(t),
                                                              np.sin(t),
                                                              color="red")
        else:
            [self.trailing_edge] = self.canvas_edges.ax2.plot([1, 1], [-1, 1], color="red")
        [self.trailing_edge_p] = self.canvas_edges.ax2.plot([0, 1], [1, 1], color="black", linestyle="--")
        [self.trailing_edge_s] = self.canvas_edges.ax2.plot([0, 1], [-1, -1], color="black", linestyle="--")

        # self.switching_widgets()

    def accept(self):
        if self.sender_name == "pushButton_impeller":
            self.parent.pump3D.impeller = self.rotor3D
        elif self.sender_name == "pushButton_inducer":
            self.parent.pump3D.inducer = self.rotor3D
        super().accept()

    def reject(self):
        super().reject()

    def update_meridonal_dict_CP(self, value):
        if self.sender().__class__ == QDoubleSpinBox:
            key_name = self.sender().objectName().replace("doubleSpinBox_", "")
        else:
            key_name = self.sender().objectName().replace("lineEdit_", "")
            value = float(value) / 1000
        self.rotor3D.meridional_dict[key_name] = value
        self.rotor3D.set_meridional_geometry()
        self.update_meridional_plot()

    def update_leading_edge(self, value):
        obj_name = self.sender().objectName().replace("doubleSpinBox_", "")
        keyname = "trailing_edge_"
        if obj_name[0] == "L":
            keyname = "leading_edge_"

        if obj_name.split("_")[-1] == "hub":
            self.rotor3D.meridional_dict[f"{keyname}@hub"] = value
        elif obj_name.split("_")[-1] == "tip":
            self.rotor3D.meridional_dict[f"{keyname}@tip"] = value
        elif obj_name.split("_")[-1] == "guide":
            self.rotor3D.meridional_dict["leading_edge_@guide"][1] = value
        else:
            self.rotor3D.meridional_dict["leading_edge_@guide"][0] = int(value - 1)
        self.rotor3D.set_meridional_geometry(category="trailing")
        self.update_meridional_plot(category="trailing")
        self.rotor3D.set_meridional_geometry(category="leading")
        self.update_meridional_plot(category="leading")

    def update_meridional_plot(self, category="all"):
        if category in ["all", "hub"]:
            self.hub_line.set_data(self.rotor3D.hub.z, self.rotor3D.hub.r)
            self.hub_node_line.set_data(self.rotor3D.hub.bezier.nodes[2], self.rotor3D.hub.bezier.nodes[0])

        if category in ["all", "tip"]:
            self.tip_line.set_data(self.rotor3D.tip.z, self.rotor3D.tip.r)
            self.tip_node_line.set_data(self.rotor3D.tip.bezier.nodes[2], self.rotor3D.tip.bezier.nodes[0])

        if category in ["all", "leading"]:
            self.leading_line.set_data(self.rotor3D.leading_edge.z, self.rotor3D.leading_edge.r)
            self.leading_node_line.set_data(self.rotor3D.leading_edge.bezier.nodes[2],
                                            self.rotor3D.leading_edge.bezier.nodes[0])

        if category in ["all", "trailing"]:
            self.trailing_line.set_data(self.rotor3D.trailing_edge.z, self.rotor3D.trailing_edge.r)

        if category in ["all", "guide"]:
            for i, guide_line in enumerate(self.guide_lines):
                guide_bezier = self.rotor3D.guides_dict["guides"][i]
                guide_line.set_data(guide_bezier.z, guide_bezier.r)
        self.canvas_meridional.ax.autoscale()
        self.canvas_meridional.draw()

    def update_beta_dict(self):
        if self.radioButton_inlet_lineer.isChecked():
            self.rotor3D.beta_dict["inlet_betas"]["method"] = "linear"
            self.rotor3D.beta_dict["inlet_betas"]["array"][0] = float(
                self.tableWidget_beta.item(0, 0).text()) * np.pi / 180
            self.rotor3D.beta_dict["inlet_betas"]["array"][-1] = float(
                self.tableWidget_beta.item(6, 0).text()) * np.pi / 180
        else:
            self.rotor3D.beta_dict["inlet_betas"]["method"] = "free"
            for row in range(self.tableWidget_beta.rowCount()):
                self.rotor3D.beta_dict["inlet_betas"]["array"][row] = float(
                    self.tableWidget_beta.item(row, 0).text()) * np.pi / 180

        if self.radioButton_outlet_lineer.isChecked():
            self.rotor3D.beta_dict["outlet_betas"]["method"] = "linear"
            self.rotor3D.beta_dict["outlet_betas"]["array"][0] = float(
                self.tableWidget_beta.item(0, 1).text()) * np.pi / 180
            self.rotor3D.beta_dict["outlet_betas"]["array"][-1] = float(
                self.tableWidget_beta.item(6, 1).text()) * np.pi / 180
        else:
            self.rotor3D.beta_dict["outlet_betas"]["method"] = "free"
            for row in range(self.tableWidget_beta.rowCount()):
                self.rotor3D.beta_dict["outlet_betas"]["array"][row] = float(
                    self.tableWidget_beta.item(row, 1).text()) * np.pi / 180

        self.rotor3D.set_beta_array()
        self.rotor3D.set_thetas_from_betas()
        self.update_beta_plot()
        self.update_table_view()

    def update_beta_dict_CP(self, value):
        key_name = self.sender().objectName().replace("doubleSpinBox_", "")
        key1 = key_name.split("_")[0]
        key2 = int(key_name.split("_")[1])
        self.rotor3D.beta_dict[key1 + "_beta_CP"][key2][0] = value
        self.rotor3D.set_beta_array()
        self.update_beta_plot()

    def update_table_view(self):
        for i in range(self.tableWidget_beta.rowCount()):
            item_inlet = self.tableWidget_beta.item(i, 0)
            item_inlet.setText(f'{self.rotor3D.beta_dict["array"][i, 0] * 180 / np.pi:.2f}')
            item_outlet = self.tableWidget_beta.item(i, 1)
            item_outlet.setText(f'{self.rotor3D.beta_dict["array"][i, -1] * 180 / np.pi:.2f}')

            if i not in [0, self.tableWidget_beta.rowCount() - 1]:
                if self.radioButton_inlet_free.isChecked():
                    item_inlet.setFlags(
                        Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                else:
                    item_inlet.setFlags(Qt.ItemIsDragEnabled | Qt.ItemIsUserCheckable)
                if self.radioButton_outlet_free.isChecked():
                    item_outlet.setFlags(
                        Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                else:
                    item_outlet.setFlags(Qt.ItemIsDragEnabled | Qt.ItemIsUserCheckable)

    def keyPressEvent(self, event):
        if self.stackedWidget.currentIndex() == 1:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self.update_beta_dict()
        else:
            pass

    def update_beta_plot(self):
        self.beta_inlet_nodes.set_data([0, 0.33, 0.67, 1],
                                       (self.rotor3D.beta_dict["array"][0, 0] + (
                                               self.rotor3D.beta_dict["array"][0, -1] - self.rotor3D.beta_dict["array"][
                                           0, 0]) * self.rotor3D.beta_cps[0, 0]) * 180 / np.pi)
        self.beta_outlet_nodes.set_data([0, 0.33, 0.67, 1],
                                        (self.rotor3D.beta_dict["array"][-1, 0] + (
                                                self.rotor3D.beta_dict["array"][-1, -1] -
                                                self.rotor3D.beta_dict["array"][
                                                    -1, 0]) * self.rotor3D.beta_cps[-1, 0]) * 180 / np.pi)

        for i, beta_line in enumerate(self.beta_lines):
            beta_line.set_ydata(self.rotor3D.beta_dict["array"][i] * 180 / np.pi)
        self.canvas_beta.ax.autoscale()
        self.canvas_beta.draw()

    def recall_1D_value(self):
        key_name = self.sender().objectName().replace("pushButton_", "")
        obj = self.rotor1D
        for key in key_name.split("_"):
            value = obj.__getattribute__(key)
            obj = value
        self.__getattribute__(f"lineEdit_{key_name}").setText(f"{value * 1000:.1f}")
        self.rotor3D.set_meridional_geometry()
        self.update_meridional_plot()

    def switching_widgets(self):
        direction = 1
        if self.sender().objectName().split("_")[-1] != "next":
            direction = -1

        index = self.stackedWidget.currentIndex()
        new_index = index + direction
        self.pushButton_next.setEnabled(True)
        self.pushButton_before.setEnabled(True)
        self.rotor3D.set_meridional_geometry()
        self.rotor3D.set_beta_array()
        self.rotor3D.set_thetas_from_betas()
        self.rotor3D.set_thickness_array()
        self.rotor3D.set_foils()
        self.rotor3D.set_leading_edge()
        self.rotor3D.set_trailing_edge()

        if new_index == 0:
            for line in (self.leading_lines + self.trailing_lines + self.blade_lines + [self.blade_hub] +
                         [self.blade_tip] + self.pressure_lines + self.suction_lines):
                line.set_linestyle("None")
            self.hub_merid.set_linestyle("-")
            self.tip_merid.set_linestyle("-")
            self.impeller_hub_inlet.set_linestyle("--")
            self.impeller_hub_exit.set_linestyle("--")
            self.impeller_tip_exit.set_linestyle("--")
            self.impeller_tip_inlet.set_linestyle("--")
            self.stackedWidget.setCurrentIndex(index + direction)
            self.pushButton_before.setEnabled(False)

        if new_index == 1:
            for line in self.leading_lines + self.trailing_lines + self.pressure_lines + self.suction_lines:
                line.set_linestyle("None")
            for line in self.blade_lines + [self.blade_hub] + [self.blade_tip]:
                line.set_linestyle("--")
            self.hub_merid.set_linestyle("-")
            self.tip_merid.set_linestyle("-")
            self.impeller_hub_inlet.set_linestyle("--")
            self.impeller_hub_exit.set_linestyle("--")
            self.impeller_tip_exit.set_linestyle("--")
            self.impeller_tip_inlet.set_linestyle("--")
            self.stackedWidget.setCurrentIndex(index + direction)

        if new_index == 2:
            for line in self.leading_lines + self.trailing_lines + self.blade_lines:
                line.set_linestyle("None")
            for line in self.pressure_lines + self.suction_lines:
                line.set_linestyle("-")
            for line in [self.blade_hub] + [self.blade_tip]:
                line.set_linestyle("--")
            self.hub_merid.set_linestyle("-")
            self.tip_merid.set_linestyle("-")
            self.impeller_hub_inlet.set_linestyle("--")
            self.impeller_hub_exit.set_linestyle("--")
            self.impeller_tip_exit.set_linestyle("--")
            self.impeller_tip_inlet.set_linestyle("--")
            self.stackedWidget.setCurrentIndex(index + direction)

        if new_index in [3, 4]:
            for line in self.blade_lines:
                line.set_linestyle("None")
            for line in self.leading_lines + self.trailing_lines + self.pressure_lines + self.suction_lines:
                line.set_linestyle("-")
            for line in [self.blade_hub] + [self.blade_tip]:
                line.set_linestyle("--")
            self.hub_merid.set_linestyle("-")
            self.tip_merid.set_linestyle("-")
            self.impeller_hub_inlet.set_linestyle("--")
            self.impeller_hub_exit.set_linestyle("--")
            self.impeller_tip_exit.set_linestyle("--")
            self.impeller_tip_inlet.set_linestyle("--")
            self.stackedWidget.setCurrentIndex(index + direction)
            if new_index == 4:
                self.pushButton_next.setEnabled(False)

        self.plot_blade()

    def update_thickness_dict(self, value):
        obj_name = self.sender().objectName().replace("doubleSpinBox_", "")
        obj_name.replace("lineEdit_", "")
        hub_or_tip = self.sender().parent().objectName().replace("groupBox_", "")
        t_dict = self.rotor3D.thickness_dict[hub_or_tip]
        if obj_name.split("_")[-1] == "piece":
            node_spin = getattr(self, "doubleSpinBox_" + hub_or_tip + "_node")
            node_spin.setMaximum(int(value))
            dummy_dict = {
                "number_of_piece": int(value),
                "nodes": list(range(int(value + 1))),
                "thickness": np.ones(int(value + 1)) * t_dict["thickness"][-1],
                "distribution": np.ones(int(value + 1)) * t_dict["distribution"][-1],
                "location": np.linspace(0, 1, int(value + 1))
            }
            self.rotor3D.thickness_dict[hub_or_tip] = dummy_dict

        if obj_name.split("_")[-1] == "node":
            loc = getattr(self, "doubleSpinBox_" + hub_or_tip + "_loc")
            loc.setValue(t_dict["location"][int(value)])
            thickness = getattr(self, "lineEdit_" + hub_or_tip + "_thickness")
            thickness.setText(f'{t_dict["thickness"][int(value)] * 1000:.1f}')
            dist = getattr(self, "doubleSpinBox_" + hub_or_tip + "_dist")
            dist.setValue(t_dict["distribution"][int(value)])
            loc.setEnabled(True)
            if int(value) in [0, int(t_dict["number_of_piece"])]:
                loc.setEnabled(False)

        node = getattr(self, "doubleSpinBox_" + hub_or_tip + "_node")
        index = node.value()
        if obj_name.split("_")[-1] == "loc":
            self.rotor3D.thickness_dict[hub_or_tip]["location"][int(index)] = value
        if obj_name.split("_")[-1] == "dist":
            self.rotor3D.thickness_dict[hub_or_tip]["distribution"][int(index)] = value
        if obj_name.split("_")[-1] == "thickness":
            self.rotor3D.thickness_dict[hub_or_tip]["thickness"][int(index)] = float(value) / 1000
        self.rotor3D.set_thickness_array()
        self.rotor3D.set_foils()
        self.update_thickness_plot()

    def update_thickness_plot(self):
        hub_node = int(self.doubleSpinBox_hub_node.value())
        tip_node = int(self.doubleSpinBox_tip_node.value())
        self.hub_pressure_line.set_data(self.rotor3D.thickness_dict["hub"]["location"],
                                        self.rotor3D.thickness_dict["hub"]["thickness"] *
                                        self.rotor3D.thickness_dict["hub"][
                                            "distribution"] * 1000)

        self.hub_suction_line.set_data(self.rotor3D.thickness_dict["hub"]["location"],
                                       -self.rotor3D.thickness_dict["hub"]["thickness"] *
                                       (1 - self.rotor3D.thickness_dict["hub"]["distribution"]) * 1000)

        self.tip_pressure_line.set_data(self.rotor3D.thickness_dict["tip"]["location"],
                                        self.rotor3D.thickness_dict["tip"]["thickness"] *
                                        self.rotor3D.thickness_dict["tip"][
                                            "distribution"] * 1000)

        self.tip_suction_line.set_data(self.rotor3D.thickness_dict["tip"]["location"],
                                       -self.rotor3D.thickness_dict["tip"]["thickness"] *
                                       (1 - self.rotor3D.thickness_dict["tip"]["distribution"]) * 1000)

        self.hub_pointer.set_offsets(
            np.transpose([self.rotor3D.thickness_dict["hub"]["location"][hub_node] * np.ones(2),
                          np.array([1, -1]) * self.rotor3D.thickness_dict["hub"]["thickness"][
                              hub_node] *
                          np.array([self.rotor3D.thickness_dict["hub"]["distribution"][hub_node],
                                    1 - self.rotor3D.thickness_dict["hub"]["distribution"][
                                        hub_node]]) * 1000]))
        self.tip_pointer.set_offsets(
            np.transpose([self.rotor3D.thickness_dict["tip"]["location"][tip_node] * np.ones(2),
                          np.array([1, -1]) * self.rotor3D.thickness_dict["tip"]["thickness"][
                              tip_node] *
                          np.array([self.rotor3D.thickness_dict["tip"]["distribution"][tip_node],
                                    1 - self.rotor3D.thickness_dict["tip"]["distribution"][
                                        tip_node]]) * 1000]))

        self.hub_hatch.remove()
        self.tip_hatch.remove()

        self.hub_hatch = self.canvas_thickness.ax.fill_between(self.rotor3D.thickness_dict["hub"]["location"],
                                                               self.rotor3D.thickness_dict["hub"]["thickness"] *
                                                               self.rotor3D.thickness_dict["hub"][
                                                                   "distribution"] * 1000,
                                                               - self.rotor3D.thickness_dict["hub"]["thickness"] * (
                                                                       1 - self.rotor3D.thickness_dict["hub"][
                                                                   "distribution"]) * 1000, interpolate=True,
                                                               color=colors.CSS4_COLORS["lightblue"], hatch="//",
                                                               edgecolor="blue")

        self.tip_hatch = self.canvas_thickness.ax.fill_between(self.rotor3D.thickness_dict["tip"]["location"],
                                                               self.rotor3D.thickness_dict["tip"]["thickness"] *
                                                               self.rotor3D.thickness_dict["tip"][
                                                                   "distribution"] * 1000,
                                                               - self.rotor3D.thickness_dict["tip"]["thickness"] *
                                                               (1 - self.rotor3D.thickness_dict["tip"][
                                                                   "distribution"]) * 1000, interpolate=True,
                                                               color=colors.CSS4_COLORS["lightgreen"], hatch="\\",
                                                               edgecolor="green")
        self.canvas_edges.ax1.margins(0.05, 0.1)
        self.canvas_edges.ax2.margins(0.05, 0.1)
        self.canvas_thickness.draw()

    def handle_comboBox_change(self, index):
        if self.sender().objectName() == "comboBox_leading":
            self.stackedWidget_leading.setCurrentIndex(index)
            self.rotor3D.leading_edge_dict["method"] = self.sender().currentText()
        elif self.sender().objectName() == "comboBox_trailing":
            self.stackedWidget_trailing.setCurrentIndex(index)
            self.rotor3D.trailing_edge_dict["method"] = self.sender().currentText()

    def update_leading_edge_dict(self):
        obj_name = self.sender().objectName()
        key_name = obj_name.split("_")[-1]
        if obj_name.split("_")[1] == "leading":
            self.rotor3D.leading_edge_dict[key_name] = self.sender().value()
            self.rotor3D.set_leading_edge()
        else:
            self.rotor3D.trailing_edge_dict[key_name] = self.sender().value()
            self.rotor3D.set_trailing_edge()
        self.update_leading_edge_plot()

    def update_leading_edge_plot(self):
        t = np.linspace(0.5 * np.pi, 1.5 * np.pi, 20)
        self.leading_edge.set_data(self.rotor3D.leading_edge_dict["ratio"] * np.cos(t), np.sin(t))
        if self.rotor3D.trailing_edge_dict["method"] == "Eliptik":
            self.trailing_edge.set_data(1 - self.rotor3D.trailing_edge_dict["ratio"] * np.cos(t), np.sin(t))
        elif self.rotor3D.trailing_edge_dict["method"] == "Çıkış Üzerinde":
            self.trailing_edge.set_data([1, 1], [1, -1])
        self.canvas_edges.draw()

    def plot_blade(self):

        self.hub_merid.set_data(self.rotor3D.hub.r, np.zeros_like(self.rotor3D.hub.r))
        self.hub_merid.set_3d_properties(self.rotor3D.hub.z)

        self.tip_merid.set_data(self.rotor3D.tip.r, np.zeros_like(self.rotor3D.tip.r))
        self.tip_merid.set_3d_properties(self.rotor3D.tip.z)

        circle = np.linspace(0, 2 * np.pi)
        self.impeller_hub_exit.set_data(self.rotor3D.hub.r[-1] * np.cos(circle),
                                        self.rotor3D.hub.r[-1] * np.sin(circle))
        self.impeller_hub_exit.set_3d_properties(self.rotor3D.hub.z[-1])
        self.impeller_hub_inlet.set_data(self.rotor3D.hub.r[0] * np.cos(circle),
                                         self.rotor3D.hub.r[0] * np.sin(circle))
        self.impeller_hub_inlet.set_3d_properties(self.rotor3D.hub.z[0])
        self.impeller_tip_exit.set_data(self.rotor3D.tip.r[-1] * np.cos(circle),
                                        self.rotor3D.tip.r[-1] * np.sin(circle))
        self.impeller_tip_exit.set_3d_properties(self.rotor3D.tip.z[-1])
        self.impeller_tip_inlet.set_data(self.rotor3D.tip.r[0] * np.cos(circle),
                                         self.rotor3D.tip.r[0] * np.sin(circle))
        self.impeller_tip_inlet.set_3d_properties(self.rotor3D.tip.z[0])

        self.blade_hub.set_data(self.rotor3D.blade_hub.x, self.rotor3D.blade_hub.y)
        self.blade_hub.set_3d_properties(self.rotor3D.blade_hub.z)

        self.blade_tip.set_data(self.rotor3D.blade_tip.x, self.rotor3D.blade_tip.y)
        self.blade_tip.set_3d_properties(self.rotor3D.blade_tip.z)

        for i, blade_line in enumerate(self.rotor3D.guides_dict["blade_guides"]):
            self.blade_lines[i].set_data(blade_line.x, blade_line.y)
            self.blade_lines[i].set_3d_properties(blade_line.z)

        for i, (pressure_line, suction_line) in enumerate(
                zip(self.rotor3D.foil_dict["pressure"]["curves"], self.rotor3D.foil_dict["suction"]["curves"])):
            self.pressure_lines[i].set_data(pressure_line.x, pressure_line.y)
            self.pressure_lines[i].set_3d_properties(pressure_line.z)
            self.suction_lines[i].set_data(suction_line.x, suction_line.y)
            self.suction_lines[i].set_3d_properties(suction_line.z)

        for i, edge_line in enumerate(self.rotor3D.leading_edge_dict["curves"]):
            self.leading_lines[i].set_data(edge_line.x, edge_line.y)
            self.leading_lines[i].set_3d_properties(edge_line.z)

        if self.rotor3D.trailing_edge_dict["method"] == "Eliptik":
            for i, edge_line in enumerate(self.rotor3D.trailing_edge_dict["curves"]):
                self.trailing_lines[i].set_data(edge_line.x, edge_line.y)
                self.trailing_lines[i].set_3d_properties(edge_line.z)

        self.canvas_blade.draw()

    def export_step(self):
        salome_path = QFileDialog.getExistingDirectory(self, "SALOME KONUMUNU SEÇ")
        if salome_path:
            file_path = QFileDialog.getSaveFileName(self, f"Dosyayı Kaydet", f"{self.rotor3D.type}.stp", ".stp")[0]
            if file_path:
                if file_path.split(".")[-1] != "stp":
                    file_path += ".stp"
                read_path = Path(os.getcwd())
                salome_script = resource_path(f"resource/salome_scripts/salome_{self.rotor3D.type}.py")
                with open(salome_script, "r") as f:
                    lines = f.readlines()
                lines[9] = 'file_path = "' + file_path + '"\n'
                lines[10] = 'read_path = "' + read_path.as_posix() + '/temp/"\n'
                with open(salome_script, "w") as f:
                    f.writelines(lines)
                self.rotor3D.export_npy()
                os.system(salome_path + f"/run_salome.bat -t {salome_script}")
        else:
            pass

    def export_dict(self):
        file_path = QFileDialog.getSaveFileName(self, f"Dosyayı Kaydet", f"{self.rotor3D.type}.json", ".json")[0]
        if file_path:
            self.rotor3D.export_as_dict()
            if file_path.split(".")[-1] != "json":
                file_path += ".json"
            with open(file_path, "w") as outfile:
                json.dump(self.rotor3D.export_dict, outfile, indent=5, cls=NumpyEncoder)
        else:
            pass

    def export_object(self):
        extension = ".imp3D"
        if self.rotor3D.type == "inducer":
            extension = ".ind3D"

        file_path = \
            QFileDialog.getSaveFileName(self, "Dosyayı Kaydet", f"{self.rotor3D.type}{extension}", f"{extension}")[0]
        if file_path:
            if file_path.split(".")[-1] != extension[1:]:
                file_path += extension
            filehandler = open(file_path, "wb")
            pickle.dump(self.rotor3D, filehandler)


class VoluteDialog(QDialog):
    def __init__(self, ui_file, parent=None, pump3D=None):
        super(VoluteDialog, self).__init__()
        self.to_export = None
        self.inputs_changed = None
        loadUi(ui_file, self)
        self.parent = parent
        self.pump3D = pump3D
        self.volute = self.parent.pump3D.volute
        self.old_volute = self.parent.pump3D.volute

        self.label_imp_hub_r.setText(f"{self.pump3D.impeller.hub.r[-1] * 1000:.2f}")
        self.label_imp_tip_r.setText(f"{self.pump3D.impeller.tip.r[-1] * 1000:.2f}")
        self.label_imp_hub_z.setText(f"{self.pump3D.impeller.hub.z[-1] * 1000:.2f}")
        self.label_imp_tip_z.setText(f"{self.pump3D.impeller.tip.z[-1] * 1000:.2f}")

        self.pushButton_set_inlet.clicked.connect(self.set_inlet_from_impeller)
        self.pushButton_cancel.clicked.connect(self.handle_cancel)
        self.pushButton_ok.clicked.connect(self.handle_ok)

        self.horizontalSlider_diameter_ratio.valueChanged.connect(self.handle_slider)
        self.horizontalSlider_inlet_width_ratio.valueChanged.connect(self.handle_slider)
        self.horizontalSlider_cone_angle.valueChanged.connect(self.handle_slider)
        self.horizontalSlider_trapezoid_angle.valueChanged.connect(self.handle_slider)
        self.horizontalSlider_wrap_angle.valueChanged.connect(self.handle_slider)

        self.comboBox_cross_type.currentTextChanged.connect(self.handle_cross_change)
        self.handle_cross_change("Trapezoid")

        self.checkBox_set_inlet_flow.toggled.connect(self.handle_lineEdit)
        self.checkBox_set_inlet_flow.toggled.connect(self.update_spiral)

        for radioButton in self.frame_eccentricity.findChildren(QRadioButton):
            radioButton.toggled.connect(self.set_diffuser_eccentricity)

        for lineEdit in self.page_inlet.findChildren(QLineEdit):
            lineEdit.setValidator(QDoubleValidator())
            att_name = lineEdit.objectName().replace('lineEdit_', '')
            if att_name.split("_")[-1] == "ratio":
                lineEdit.setText(f"{getattr(self.volute, att_name):.0f}")
            else:
                lineEdit.setText(f"{getattr(self.volute, att_name) * 1000:.1f}")
            lineEdit.textChanged.connect(self.handle_lineEdit)
            lineEdit.editingFinished.connect(self.update_inlet)

        for lineEdit in self.page_cross.findChildren(QLineEdit):
            lineEdit.setValidator(QDoubleValidator())
            att_name = lineEdit.objectName().replace('lineEdit_', '')
            key_name = lineEdit.parent().objectName().replace("frame_", "")
            if att_name.split("_")[-1] == "angle":
                lineEdit.setText(f"{self.volute.cross_section[key_name][att_name] * 180 / np.pi:.0f}")
            else:
                lineEdit.setText(f"{self.volute.cross_section[key_name][att_name] * 1000:.1f}")
            lineEdit.textChanged.connect(self.handle_lineEdit)
            lineEdit.editingFinished.connect(self.update_cross_section)

        for frame in [self.frame_radius_based, self.frame_trapezoid]:
            if frame.objectName() not in [f"frame_{self.volute.cross_section_type}", "frame_type"]:
                frame.hide()

        for lineEdit in self.page_spiral.findChildren(QLineEdit):
            lineEdit.setValidator(QDoubleValidator())
            att_name = lineEdit.objectName().replace('lineEdit_', '')
            if att_name.split("_")[-1] == "flow":
                lineEdit.setText(f"{getattr(self.volute, att_name) * 1000:.2f}")
            else:
                lineEdit.setText(f"{getattr(self.volute, att_name):.1f}")
            lineEdit.textChanged.connect(self.handle_lineEdit)
            lineEdit.editingFinished.connect(self.update_spiral)

        for lineEdit in self.page_diffuser.findChildren(QLineEdit):
            lineEdit.setValidator(QDoubleValidator())
            att_name = lineEdit.objectName().replace('lineEdit_diffuser_', '')
            lineEdit.setText(f"{getattr(self.volute.exit_diff, att_name) * 1000:.1f}")
            lineEdit.textChanged.connect(self.handle_lineEdit)
            lineEdit.editingFinished.connect(self.update_plots)
            lineEdit.editingFinished.connect(self.update_3d)

        for lineEdit in self.page_cutwater.findChildren(QLineEdit):
            lineEdit.setValidator(QDoubleValidator())
            att_name = lineEdit.objectName().replace('lineEdit_cutwater_', '')
            lineEdit.setText(f"{getattr(self.volute.cut_water, att_name):.3f}")
            lineEdit.textChanged.connect(self.handle_lineEdit)
            lineEdit.editingFinished.connect(self.update_plots)
            lineEdit.editingFinished.connect(self.update_3d)

        self.toolBox.currentChanged.connect(self.switch_plots)
        # Viewers
        self.viewer_inlet = GeometryPlot(self, got_axline=False)
        self.gridLayout_upper_1.addWidget(self.viewer_inlet)

        self.viewer_cross = GeometryPlot(self, got_axline=False)
        self.gridLayout_lower_1.addWidget(self.viewer_cross)

        self.viewer_spiral = GeometryPlot(self, rz=False)
        self.gridLayout_upper_2.addWidget(self.viewer_spiral)

        self.viewer_diffuser = GeometryPlot(self, rz=False, got_axline=False)
        self.gridLayout_lower_2.addWidget(self.viewer_diffuser)

        self.viewer_cutwater = GeometryPlot(self, rz=False, got_axline=False)
        self.gridLayout_lower_3.addWidget(self.viewer_cutwater)

        self.line_impeller_outlet = Line2D(xdata=[], ydata=[], color="gray", lw=1.5, label="Volute Inlet")
        self.line_inlet = Line2D(xdata=[], ydata=[], color="red", lw=1, label="Impeller Outlet")
        self.lines_cross_section = []

        self.viewer_inlet.ax.add_line(self.line_impeller_outlet)
        self.viewer_inlet.ax.add_line(self.line_inlet)
        self.viewer_inlet.ax.legend()

        for i in range(0, 365, 15):
            if i == 360:
                self.lines_cross_section.append(Line2D(xdata=[], ydata=[], color="blue", lw=2, label="360°"))
            elif i == 0:
                self.lines_cross_section.append(Line2D(xdata=[], ydata=[], color="red", lw=2, label="0°"))
            else:
                self.lines_cross_section.append(Line2D(xdata=[], ydata=[], color="gray", lw=0.5))
            self.viewer_cross.ax.add_line(self.lines_cross_section[-1])
        self.viewer_cross.ax.legend()

        self.line_spiral = Line2D(xdata=[], ydata=[], color="red", label="R Maximum")
        self.circle_inlet = Line2D(xdata=[], ydata=[], color="black", label="Volute Inlet")
        self.line_diffuser_inner = Line2D(xdata=[], ydata=[], color="red", label="Diffuser X Minimum")
        self.line_diffuser_outer = Line2D(xdata=[], ydata=[], color="red", label="Diffuser X Maximum")
        self.line_diffuser_center = Line2D(xdata=[], ydata=[], color="blue", linestyle='-.', label="Diffuser Center")
        self.line_cutwater_top = Line2D(xdata=[], ydata=[], color="blue", alpha=0.5, label="Cut-Water Top")
        self.line_cutwater_mid = Line2D(xdata=[], ydata=[], color="red", label="Cut-Water Mid")
        self.line_cutwater_bottom = Line2D(xdata=[], ydata=[], color="green", alpha=0.5, label="Cut-Water Bottom")
        self.line_cutwater_diff = Line2D(xdata=[], ydata=[], color="gray", label="Diffuser Edge")
        self.line_cutwater_vol = Line2D(xdata=[], ydata=[], color="gray", label="Volute Edge")

        self.viewer_spiral.ax.add_line(self.line_spiral)
        self.viewer_spiral.ax.add_line(self.circle_inlet)
        self.viewer_diffuser.ax.add_line(self.line_diffuser_inner)
        self.viewer_diffuser.ax.add_line(self.line_diffuser_outer)
        self.viewer_diffuser.ax.add_line(self.line_diffuser_center)
        self.viewer_cutwater.ax.add_line(self.line_cutwater_top)
        self.viewer_cutwater.ax.add_line(self.line_cutwater_mid)
        self.viewer_cutwater.ax.add_line(self.line_cutwater_bottom)
        self.viewer_cutwater.ax.add_line(self.line_cutwater_diff)
        self.viewer_cutwater.ax.add_line(self.line_cutwater_vol)
        self.viewer_spiral.ax.legend()
        self.viewer_diffuser.ax.legend()
        self.viewer_cutwater.ax.legend()

        self.viewer_3D = MayaviQWidget(self)
        self.gridLayout_3d.addWidget(self.viewer_3D)

        self.spiral_geo = PointCloudGeometry()
        self.diffuser_geo = PointCloudGeometry()
        self.cut_water_geo = PointCloudGeometry()

        self.cutwater_patch_top_volute = PointCloudLine()
        self.cutwater_patch_top_diffuser = PointCloudLine()
        self.cutwater_patch_top_arc = PointCloudLine()
        self.cutwater_patch_bottom_volute = PointCloudLine()
        self.cutwater_patch_bottom_diffuser = PointCloudLine()
        self.cutwater_patch_bottom_arc = PointCloudLine()

        for i in range(len(self.volute.sections[0])):
            surf = []
            for sec in self.volute.sections:
                surf.append(sec[i])
            self.spiral_geo.surfaces.append(PointCloudSurface(surf))
        self.diffuser_geo.surfaces = [PointCloudSurface(self.volute.exit_diff.all_sections.tolist())]
        self.cut_water_geo.surfaces = [PointCloudSurface(self.volute.cut_water.surface.tolist())]
        self.cutwater_patch_top_volute.points = self.volute.cut_water.filler_top_curve2
        self.cutwater_patch_top_diffuser.points = self.volute.cut_water.filler_top_curve1
        self.cutwater_patch_top_arc.points = self.volute.cut_water.filler_top_curve3
        self.cutwater_patch_bottom_volute.points = self.volute.cut_water.filler_bottom_curve2
        self.cutwater_patch_bottom_diffuser.points = self.volute.cut_water.filler_bottom_curve1
        self.cutwater_patch_bottom_arc.points = self.volute.cut_water.filler_bottom_curve3

        self.viewer_3D.add_geometry(self.spiral_geo)
        self.viewer_3D.add_geometry(self.diffuser_geo)
        self.viewer_3D.add_geometry(self.cut_water_geo)
        self.viewer_3D.add_line(self.cutwater_patch_top_volute)
        self.viewer_3D.add_line(self.cutwater_patch_top_diffuser)
        self.viewer_3D.add_line(self.cutwater_patch_top_arc)
        self.viewer_3D.add_line(self.cutwater_patch_bottom_volute)
        self.viewer_3D.add_line(self.cutwater_patch_bottom_diffuser)
        self.viewer_3D.add_line(self.cutwater_patch_bottom_arc)

        self.update_plots(0)
        self.switch_plots(0)

    def update_3d(self, index=None):
        if index == None:
            index = self.toolBox.currentIndex()
        if index in [0, 1]:
            pass
        elif index == 2:
            for i in range(len(self.volute.sections[0])):
                surf = []
                for sec in self.volute.sections:
                    surf.append(sec[i])
                self.spiral_geo.surfaces[i].curves = surf
            self.spiral_geo.set_opacity(1)
            self.diffuser_geo.set_opacity(0)
            self.cut_water_geo.set_opacity(0)
            self.cutwater_patch_top_volute.set_opacity(0)
            self.cutwater_patch_top_diffuser.set_opacity(0)
            self.cutwater_patch_top_arc.set_opacity(0)
            self.cutwater_patch_bottom_volute.set_opacity(0)
            self.cutwater_patch_bottom_diffuser.set_opacity(0)
            self.cutwater_patch_bottom_arc.set_opacity(0)
            self.spiral_geo.set_color((1, 1, 1))
            self.diffuser_geo.set_color((70 / 255, 70 / 255, 70 / 255))
            self.cut_water_geo.set_color((70 / 255, 70 / 255, 70 / 255))

        elif index == 3:
            self.diffuser_geo.surfaces[0].curves = self.volute.exit_diff.all_sections.tolist()
            self.spiral_geo.set_opacity(0.5)
            self.diffuser_geo.set_opacity(1)
            self.cut_water_geo.set_opacity(0)
            self.cutwater_patch_top_volute.set_opacity(0)
            self.cutwater_patch_top_diffuser.set_opacity(0)
            self.cutwater_patch_top_arc.set_opacity(0)
            self.cutwater_patch_bottom_volute.set_opacity(0)
            self.cutwater_patch_bottom_diffuser.set_opacity(0)
            self.cutwater_patch_bottom_arc.set_opacity(0)
            self.diffuser_geo.set_color((1, 1, 1))
            self.spiral_geo.set_color((70 / 255, 70 / 255, 70 / 255))
            self.cut_water_geo.set_color((70 / 255, 70 / 255, 70 / 255))
        elif index == 4:
            self.cut_water_geo.surfaces[0].curves = self.volute.cut_water.surface.tolist()
            self.cutwater_patch_top_volute.points = self.volute.cut_water.filler_top_curve1
            self.cutwater_patch_top_diffuser.points = self.volute.cut_water.filler_top_curve2
            self.cutwater_patch_top_arc.points = self.volute.cut_water.filler_top_curve3
            self.cutwater_patch_bottom_volute.points = self.volute.cut_water.filler_bottom_curve1
            self.cutwater_patch_bottom_diffuser.points = self.volute.cut_water.filler_bottom_curve2
            self.cutwater_patch_bottom_arc.points = self.volute.cut_water.filler_bottom_curve3
            self.spiral_geo.set_opacity(0.5)
            self.diffuser_geo.set_opacity(0.5)
            self.cut_water_geo.set_opacity(1)
            self.cutwater_patch_top_volute.set_opacity(1)
            self.cutwater_patch_top_diffuser.set_opacity(1)
            self.cutwater_patch_top_arc.set_opacity(1)
            self.cutwater_patch_bottom_volute.set_opacity(1)
            self.cutwater_patch_bottom_diffuser.set_opacity(1)
            self.cutwater_patch_bottom_arc.set_opacity(1)
            self.cut_water_geo.set_color((1, 1, 1))
            self.diffuser_geo.set_color((70 / 255, 70 / 255, 70 / 255))
            self.spiral_geo.set_color((70 / 255, 70 / 255, 70 / 255))

        self.viewer_3D.update_viewer()

    def update_plots(self, index=None):
        if index == None:
            index = self.toolBox.currentIndex()
        if index == 0:
            self.line_inlet.set_data([(- 0.5 * self.volute.impeller_width - self.volute.offset_tip_z) * 1000,
                                      (+ 0.5 * self.volute.impeller_width + self.volute.offset_hub_z) * 1000],
                                     [(self.volute.absolute_r + self.volute.offset_r) * 1000] * 2)

            self.line_impeller_outlet.set_data([- 0.5 * self.volute.impeller_width * 1000,
                                                + 0.5 * self.volute.impeller_width * 1000],
                                               [self.volute.anchor_r * 1000] * 2)

            self.viewer_inlet.update_plot()
        if index in [0, 1, 2]:
            for i, angle in enumerate(range(0, 365, 15)):
                sec = self.volute.sections[angle]
                curve_r_conc = np.concatenate(tuple(curve.r for curve in sec))
                curve_z_conc = np.concatenate(tuple(curve.z for curve in sec))
                self.lines_cross_section[i].set_data(curve_z_conc * 1000, curve_r_conc * 1000)
            self.viewer_cross.update_plot()

        if index in [2, 3]:
            self.line_spiral.set_data(
                [np.max([curve.r * 1000 for curve in sec]) * np.cos(sec[0].theta) for sec in self.volute.sections],
                [np.max([curve.r * 1000 for curve in sec]) * np.sin(sec[0].theta) for sec in self.volute.sections])

            self.circle_inlet.set_data(
                [np.min([curve.r * 1000 for curve in sec]) * np.cos(sec[0].theta) for sec in self.volute.sections],
                [np.min([curve.r * 1000 for curve in sec]) * np.sin(sec[0].theta) for sec in self.volute.sections])
            self.viewer_spiral.update_plot()

        if index in [3, 4]:
            self.line_diffuser_inner.set_data(
                np.min(self.volute.exit_diff.all_sections[:, :, 0], axis=1) * 1000,
                [sec[np.argmin(sec[0]), 1] * 1000 for sec in self.volute.exit_diff.all_sections])

            self.line_diffuser_outer.set_data(
                np.max(self.volute.exit_diff.all_sections[:, :, 0], axis=1) * 1000,
                [sec[np.argmax(sec[0]), 1] * 1000 for sec in self.volute.exit_diff.all_sections])

            self.line_diffuser_center.set_data(self.volute.exit_diff.center_line[:, 0] * 1000,
                                               self.volute.exit_diff.center_line[:, 1] * 1000)

            self.viewer_diffuser.update_plot()

        if index == 4:
            self.line_cutwater_top.set_data(self.volute.cut_water.surface[0, :, 0] * 1000,
                                            self.volute.cut_water.surface[0, :, 1] * 1000)

            mid_indx = len(self.volute.cut_water.surface) // 2
            self.line_cutwater_mid.set_data(self.volute.cut_water.surface[mid_indx, :, 0] * 1000,
                                            self.volute.cut_water.surface[mid_indx, :, 1] * 1000)

            self.line_cutwater_bottom.set_data(self.volute.cut_water.surface[-1, :, 0] * 1000,
                                               self.volute.cut_water.surface[-1, :, 1] * 1000)

            index = int(self.volute.cut_water.start_angle)
            self.line_cutwater_vol.set_data(
                [np.max([curve.r * 1000 for curve in sec]) * np.cos(sec[0].theta) for sec in
                 self.volute.sections[index - 10:index + 10]],
                [np.max([curve.r * 1000 for curve in sec]) * np.sin(sec[0].theta) for sec in
                 self.volute.sections[index - 10:index + 10]])
            y_max = np.max(self.volute.cut_water.edge_volute[:, 1])
            diff_slice_idx = np.argmin(np.abs(self.volute.exit_diff.all_sections[:, 0, 1] - y_max))

            index_dif = slice(diff_slice_idx - 5, diff_slice_idx + 5)
            self.line_cutwater_diff.set_data(
                np.min(self.volute.exit_diff.all_sections[index_dif, :, 0], axis=1) * 1000,
                [sec[np.argmin(sec[0]), 1] * 1000 for sec in self.volute.exit_diff.all_sections[index_dif]])
            self.viewer_cutwater.update_plot()

    def switch_plots(self, index):
        if index == 0:
            self.stackedWidget_upper.setCurrentIndex(0)
            self.stackedWidget_lower.setCurrentIndex(0)
            self.dockWidget_rz.hide()
            self.dockWidget_3d.hide()
        elif index == 1:
            self.update_cross_section()
            self.stackedWidget_upper.setCurrentIndex(0)
            self.stackedWidget_lower.setCurrentIndex(1)
            self.dockWidget_rz.show()
            self.dockWidget_3d.hide()
        elif index == 2:
            self.update_spiral()
            self.stackedWidget_upper.setCurrentIndex(1)
            self.stackedWidget_lower.setCurrentIndex(1)
            self.dockWidget_rz.show()
            self.dockWidget_3d.show()
        elif index == 3:
            self.volute.exit_diff.update_surface()
            self.stackedWidget_upper.setCurrentIndex(1)
            self.stackedWidget_lower.setCurrentIndex(2)
            self.dockWidget_rz.show()
            self.dockWidget_3d.show()
        elif index == 4:
            self.stackedWidget_upper.setCurrentIndex(1)
            self.stackedWidget_lower.setCurrentIndex(3)
            self.dockWidget_rz.show()
            self.dockWidget_3d.show()

        self.update_plots(index)
        self.update_3d(index)
        pass

    def set_diffuser_eccentricity(self):
        att_name = self.sender().objectName().replace("radioButton_eccentricity_", "")
        if att_name == "pos":
            self.volute.exit_diff.eccentricity_mode = 1
        elif att_name == "neg":
            self.volute.exit_diff.eccentricity_mode = -1
        elif att_name == "zero":
            self.volute.exit_diff.eccentricity_mode = 0
        else:
            self.volute.exit_diff.eccentricity_mode = None
            self.volute.exit_diff.update_surface()
        self.update_plots(3)
        self.update_3d(3)

    def set_inlet_impeller_flow(self):
        if self.checkBox_set_inlet_flow.isChecked():
            self.volute.blind_vol_flow = 0
            self.volute.use_imp_inlet_flow = True
        else:
            self.volute.use_imp_inlet_flow = False
            self.volute.ref_inlet_cu = float(self.lineEdit_ref_inlet_cu.text())
            self.volute.ref_vol_flow = float(self.lineEdit_ref_vol_flow.text()) / 1000
            self.volute.blind_vol_flow = float(self.lineEdit_blind_vol_flow.text()) / 1000

        self.update_plots(0)
        self.update_3d(0)

    def update_spiral(self):
        self.volute.set_wrap_angle()
        for lineEdit in self.page_spiral.findChildren(QLineEdit):
            att_name = lineEdit.objectName().replace("lineEdit_", "")
            lineEdit.blockSignals(True)
            if att_name.split("_")[-1] == "flow":
                lineEdit.setText(f"{getattr(self.volute, att_name) * 1000:.2f}")
            else:
                lineEdit.setText(f"{getattr(self.volute, att_name):.1f}")
            lineEdit.blockSignals(False)
        self.update_plots(2)
        self.update_3d(2)

    def update_cross_section(self):
        self.volute.calc_sections()
        for lineEdit in self.page_cross.findChildren(QLineEdit):
            att_name = lineEdit.objectName().replace("lineEdit_", "")
            key_name = lineEdit.parent().objectName().replace("frame_", "")
            lineEdit.blockSignals(True)
            if att_name.split("_")[-1] == "angle":
                lineEdit.setText(f"{self.volute.cross_section[key_name][att_name] * 180 / np.pi:.0f}")
            else:
                lineEdit.setText(f"{self.volute.cross_section[key_name][att_name] * 1000:.0f}")
            lineEdit.blockSignals(False)
        self.update_3d(1)
        self.update_plots(1)

    def handle_cross_change(self, cross_type: str):
        getattr(self, f"frame_{self.volute.cross_section_type}").hide()
        cross_type_name = '_'.join(cross_type.split(' ')).lower()
        frame = getattr(self, f"frame_{cross_type_name}")
        self.volute.cross_section_type = cross_type_name
        if self.volute.cross_section_type == "trapezoid":
            self.lineEdit_cutwater_edge_volute_t_from.setEnabled(False)
            self.lineEdit_cutwater_edge_volute_t_to.setEnabled(False)
        else:
            self.lineEdit_cutwater_edge_volute_t_from.setEnabled(True)
            self.lineEdit_cutwater_edge_volute_t_to.setEnabled(True)
        frame.show()

    def handle_slider(self, value):
        if self.sender().objectName() in ["horizontalSlider_cone_angle", "horizontalSlider_trapezoid_angle"]:
            key_name = self.sender().objectName().replace("horizontalSlider_", "")
            self.volute.cross_section[self.volute.cross_section_type][key_name] = float(value) * np.pi / 180
            self.update_cross_section()
        elif self.sender().objectName() == "horizontalSlider_wrap_angle":
            att_name = self.sender().objectName().replace("horizontalSlider_", "")
            setattr(self.volute, att_name, float(value))
            self.update_spiral()
        else:
            att_name = self.sender().objectName().replace("horizontalSlider_", "")
            setattr(self.volute, att_name, float(value) / 100)
            self.update_inlet()

    def handle_lineEdit(self, value):
        att_name = self.sender().objectName().replace("lineEdit_", "")
        if self.toolBox.currentIndex() == 1:
            key_name = '_'.join(self.comboBox_cross_type.currentText().split(' ')).lower()
            if att_name.split("_")[-1] == "angle":
                self.volute.cross_section[key_name][att_name] = float(value) * np.pi / 180
            else:
                self.volute.cross_section[key_name][att_name] = float(value) / 1000
        elif self.toolBox.currentIndex() == 2:
            if att_name.split("_")[-1] == "flow":
                setattr(self.volute, att_name, float(value) / 1000)
            else:
                setattr(self.volute, att_name, float(value))
        elif self.toolBox.currentIndex() == 3:
            att_name = self.sender().objectName().replace("lineEdit_diffuser_", "")
            setattr(self.volute.exit_diff, att_name, float(value) / 1000)
            self.volute.exit_diff.update_surface()
        elif self.toolBox.currentIndex() == 4:
            att_name = self.sender().objectName().replace("lineEdit_cutwater_", "")
            setattr(self.volute.cut_water, att_name, float(value))
        else:
            if att_name.split("_")[-1] == "ratio":
                setattr(self.volute, att_name, float(value) * 100)
            else:
                setattr(self.volute, att_name, float(value) / 1000)

    def update_inlet(self):
        for lineEdit in self.page_inlet.findChildren(QLineEdit):
            att_name = lineEdit.objectName().replace("lineEdit_", "")
            lineEdit.blockSignals(True)
            if att_name.split("_")[-1] == "ratio":
                lineEdit.setText(f"{getattr(self.volute, att_name) * 100:.0f}")
            else:
                lineEdit.setText(f"{getattr(self.volute, att_name) * 1000:.1f}")
            lineEdit.blockSignals(False)
        self.update_plots(0)
        self.update_3d(0)

    def set_inlet_from_impeller(self):
        self.volute.absolute_r = self.pump3D.impeller.hub.r[-1]
        self.volute.absolute_hub_z = self.pump3D.impeller.tip.z[-1] + self.volute.impeller_width
        self.volute.absolute_tip_z = self.pump3D.impeller.tip.z[-1]
        self.update_inlet()

    def handle_ok(self):
        self.parent.pump3D.volute.import_dict(self.volute.export_dict())
        self.accept()

    def handle_cancel(self):
        reply = QMessageBox.question(self, 'Changes will be deleted.',
                                     'Do you want to delete?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.reject()
        else:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    window = PotaMain()
    window.show()
    app.exec()

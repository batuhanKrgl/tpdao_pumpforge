from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QLabel, QSpinBox, QDoubleSpinBox, QWidget

from apps.PumpForge3D.meridional.defaults import default_params
from apps.PumpForge3D.meridional.section_model import (
    MeridionalBezierParams,
    MeridionalSection2D,
    build_section,
)
from apps.PumpForge3D.models.meridional_models import (
    MeridionalSectionState,
    serialize_section,
)
from apps.PumpForge3D.gui.meridional_plot import MeridionalPlotWidget


@dataclass
class MeridionalControllerState:
    params: MeridionalBezierParams
    section: MeridionalSection2D


class MeridionalController(QObject):
    def __init__(
        self,
        dialog: QWidget,
        plot_widget: MeridionalPlotWidget,
        status_label: QLabel,
    ) -> None:
        super().__init__(dialog)
        self._dialog = dialog
        self._plot_widget = plot_widget
        self._status_label = status_label
        self._state: MeridionalSectionState | None = None
        self._last_valid_section: MeridionalSection2D | None = None

        self._spinboxes = {
            "sb_Lz": self._get_spinbox(QDoubleSpinBox, "sb_Lz"),
            "sb_hub_r_in": self._get_spinbox(QDoubleSpinBox, "sb_hub_r_in"),
            "sb_hub_r_out": self._get_spinbox(QDoubleSpinBox, "sb_hub_r_out"),
            "sb_b_in": self._get_spinbox(QDoubleSpinBox, "sb_b_in"),
            "sb_b_out": self._get_spinbox(QDoubleSpinBox, "sb_b_out"),
            "sb_hub_theta_in": self._get_spinbox(QDoubleSpinBox, "sb_hub_theta_in"),
            "sb_hub_theta_out": self._get_spinbox(QDoubleSpinBox, "sb_hub_theta_out"),
            "sb_tip_theta_in": self._get_spinbox(QDoubleSpinBox, "sb_tip_theta_in"),
            "sb_tip_theta_out": self._get_spinbox(QDoubleSpinBox, "sb_tip_theta_out"),
            "sb_hub_handle_in": self._get_spinbox(QDoubleSpinBox, "sb_hub_handle_in"),
            "sb_hub_handle_out": self._get_spinbox(QDoubleSpinBox, "sb_hub_handle_out"),
            "sb_tip_handle_in": self._get_spinbox(QDoubleSpinBox, "sb_tip_handle_in"),
            "sb_tip_handle_out": self._get_spinbox(QDoubleSpinBox, "sb_tip_handle_out"),
            "sb_hub_p2_z_norm": self._get_spinbox(QDoubleSpinBox, "sb_hub_p2_z_norm"),
            "sb_hub_p2_r_norm": self._get_spinbox(QDoubleSpinBox, "sb_hub_p2_r_norm"),
            "sb_tip_p2_z_norm": self._get_spinbox(QDoubleSpinBox, "sb_tip_p2_z_norm"),
            "sb_tip_p2_r_norm": self._get_spinbox(QDoubleSpinBox, "sb_tip_p2_r_norm"),
            "sb_le_ctrl_z_offset": self._get_spinbox(QDoubleSpinBox, "sb_le_ctrl_z_offset"),
            "sb_te_ctrl_z_offset": self._get_spinbox(QDoubleSpinBox, "sb_te_ctrl_z_offset"),
            "sb_le_hub_t_norm": self._get_spinbox(QDoubleSpinBox, "sb_le_hub_t_norm"),
            "sb_le_tip_t_norm": self._get_spinbox(QDoubleSpinBox, "sb_le_tip_t_norm"),
            "sb_te_hub_t_norm": self._get_spinbox(QDoubleSpinBox, "sb_te_hub_t_norm"),
            "sb_te_tip_t_norm": self._get_spinbox(QDoubleSpinBox, "sb_te_tip_t_norm"),
            "sb_n_curve_points": self._get_spinbox(QSpinBox, "sb_n_curve_points"),
        }

    def _get_spinbox(self, klass: type, name: str):
        widget = self._dialog.findChild(klass, name)
        if widget is None:
            raise ValueError(f"Missing widget '{name}' in UI.")
        return widget

    def apply_defaults(self, params: MeridionalBezierParams | None = None) -> None:
        params = params or default_params()
        self._spinboxes["sb_Lz"].setValue(self._m_to_mm(params.Lz_m))
        self._spinboxes["sb_hub_r_in"].setValue(self._m_to_mm(params.hub_r_in_m))
        self._spinboxes["sb_hub_r_out"].setValue(self._m_to_mm(params.hub_r_out_m))
        self._spinboxes["sb_b_in"].setValue(self._m_to_mm(params.b_in_m))
        self._spinboxes["sb_b_out"].setValue(self._m_to_mm(params.b_out_m))
        self._spinboxes["sb_hub_theta_in"].setValue(params.hub_theta_in_deg)
        self._spinboxes["sb_hub_theta_out"].setValue(params.hub_theta_out_deg)
        self._spinboxes["sb_tip_theta_in"].setValue(params.tip_theta_in_deg)
        self._spinboxes["sb_tip_theta_out"].setValue(params.tip_theta_out_deg)
        self._spinboxes["sb_hub_handle_in"].setValue(params.hub_handle_in)
        self._spinboxes["sb_hub_handle_out"].setValue(params.hub_handle_out)
        self._spinboxes["sb_tip_handle_in"].setValue(params.tip_handle_in)
        self._spinboxes["sb_tip_handle_out"].setValue(params.tip_handle_out)
        self._spinboxes["sb_hub_p2_z_norm"].setValue(params.hub_p2_z_norm)
        self._spinboxes["sb_hub_p2_r_norm"].setValue(params.hub_p2_r_norm)
        self._spinboxes["sb_tip_p2_z_norm"].setValue(params.tip_p2_z_norm)
        self._spinboxes["sb_tip_p2_r_norm"].setValue(params.tip_p2_r_norm)
        self._spinboxes["sb_le_ctrl_z_offset"].setValue(self._m_to_mm(params.le_ctrl_z_offset_m))
        self._spinboxes["sb_te_ctrl_z_offset"].setValue(self._m_to_mm(params.te_ctrl_z_offset_m))
        self._spinboxes["sb_le_hub_t_norm"].setValue(params.le_hub_t_norm)
        self._spinboxes["sb_le_tip_t_norm"].setValue(params.le_tip_t_norm)
        self._spinboxes["sb_te_hub_t_norm"].setValue(params.te_hub_t_norm)
        self._spinboxes["sb_te_tip_t_norm"].setValue(params.te_tip_t_norm)
        self._spinboxes["sb_n_curve_points"].setValue(params.n_curve_points)

    def connect_signals(self) -> None:
        for widget in self._spinboxes.values():
            widget.valueChanged.connect(self.on_any_parameter_changed)

    def _read_params(self) -> MeridionalBezierParams:
        return MeridionalBezierParams(
            Lz_m=self._mm_to_m(self._spinboxes["sb_Lz"].value()),
            hub_r_in_m=self._mm_to_m(self._spinboxes["sb_hub_r_in"].value()),
            hub_r_out_m=self._mm_to_m(self._spinboxes["sb_hub_r_out"].value()),
            b_in_m=self._mm_to_m(self._spinboxes["sb_b_in"].value()),
            b_out_m=self._mm_to_m(self._spinboxes["sb_b_out"].value()),
            hub_theta_in_deg=self._spinboxes["sb_hub_theta_in"].value(),
            hub_theta_out_deg=self._spinboxes["sb_hub_theta_out"].value(),
            tip_theta_in_deg=self._spinboxes["sb_tip_theta_in"].value(),
            tip_theta_out_deg=self._spinboxes["sb_tip_theta_out"].value(),
            hub_handle_in=self._spinboxes["sb_hub_handle_in"].value(),
            hub_handle_out=self._spinboxes["sb_hub_handle_out"].value(),
            tip_handle_in=self._spinboxes["sb_tip_handle_in"].value(),
            tip_handle_out=self._spinboxes["sb_tip_handle_out"].value(),
            hub_p2_z_norm=self._spinboxes["sb_hub_p2_z_norm"].value(),
            hub_p2_r_norm=self._spinboxes["sb_hub_p2_r_norm"].value(),
            tip_p2_z_norm=self._spinboxes["sb_tip_p2_z_norm"].value(),
            tip_p2_r_norm=self._spinboxes["sb_tip_p2_r_norm"].value(),
            le_ctrl_z_offset_m=self._mm_to_m(
                self._spinboxes["sb_le_ctrl_z_offset"].value()
            ),
            te_ctrl_z_offset_m=self._mm_to_m(
                self._spinboxes["sb_te_ctrl_z_offset"].value()
            ),
            le_hub_t_norm=self._spinboxes["sb_le_hub_t_norm"].value(),
            le_tip_t_norm=self._spinboxes["sb_le_tip_t_norm"].value(),
            te_hub_t_norm=self._spinboxes["sb_te_hub_t_norm"].value(),
            te_tip_t_norm=self._spinboxes["sb_te_tip_t_norm"].value(),
            n_curve_points=self._spinboxes["sb_n_curve_points"].value(),
        )

    def on_any_parameter_changed(self) -> None:
        try:
            params = self._read_params()
            section = build_section(params)
        except ValueError as exc:
            if self._last_valid_section is not None:
                self._plot_widget.plot_section(self._last_valid_section)
            self._status_label.setText(f"Meridional status: {exc}")
            return

        self._plot_widget.plot_section(section)
        self._last_valid_section = section
        self._status_label.setText("Meridional status: updated")
        self._state = MeridionalSectionState(
            params=params,
            section=section,
            section_serialized=serialize_section(section),
        )

    @staticmethod
    def _mm_to_m(value_mm: float) -> float:
        return value_mm / 1000.0

    @staticmethod
    def _m_to_mm(value_m: float) -> float:
        return value_m * 1000.0

    @property
    def state(self) -> MeridionalSectionState | None:
        return self._state

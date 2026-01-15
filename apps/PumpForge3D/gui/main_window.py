from __future__ import annotations

from pathlib import Path

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget

from apps.PumpForge3D.models.inducer3d_models import Inducer3DInputs, Inducer3DState
from apps.PumpForge3D.gui.meridional_controller import MeridionalController
from apps.PumpForge3D.gui.meridional_plot import MeridionalPlotWidget


class PumpForge3DMainWindow:
    def __init__(self) -> None:
        self._dialog: QDialog = loadUi(self._ui_path())
        self.state = self._init_state_defaults()
        self._meridional_controller: MeridionalController | None = None
        self._init_meridional_tab()

    @staticmethod
    def _ui_path() -> str:
        return str(Path(__file__).resolve().parent.parent / "ui" / "PumpForge3D-Inducer.ui")

    @staticmethod
    def _init_state_defaults() -> Inducer3DState:
        return Inducer3DState(inputs=Inducer3DInputs())

    def _init_meridional_tab(self) -> None:
        plot_container = self._dialog.findChild(QWidget, "meridionalPlotContainer")
        status_label = self._dialog.findChild(QLabel, "meridionalStatusLabel")
        if plot_container is None or status_label is None:
            return

        plot_layout = plot_container.layout()
        if plot_layout is None:
            plot_layout = QVBoxLayout(plot_container)
            plot_layout.setContentsMargins(0, 0, 0, 0)

        plot_widget = MeridionalPlotWidget(plot_container)
        plot_layout.addWidget(plot_widget)

        controller = MeridionalController(self._dialog, plot_widget, status_label)
        controller.apply_defaults()
        controller.connect_signals()
        controller.on_any_parameter_changed()
        self._meridional_controller = controller

    def _placeholder_future_methods(self) -> None:
        return None

    def show(self) -> None:
        self._dialog.show()

    def close(self) -> None:
        self._dialog.close()

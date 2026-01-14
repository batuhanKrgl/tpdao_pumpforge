from __future__ import annotations

from pathlib import Path

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow

from apps.PumpForge3D.models.inducer3d_models import Inducer3DInputs, Inducer3DState


class PumpForge3DMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        loadUi(self._ui_path(), self)
        self.state = self._init_state_defaults()

    @staticmethod
    def _ui_path() -> str:
        return str(Path(__file__).resolve().parent.parent / "ui" / "PumpForge3D-Inducer.ui")

    @staticmethod
    def _init_state_defaults() -> Inducer3DState:
        return Inducer3DState(inputs=Inducer3DInputs())

    def _placeholder_future_methods(self) -> None:
        return None

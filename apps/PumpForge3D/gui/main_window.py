from __future__ import annotations

from pathlib import Path

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog

from apps.PumpForge3D.models.inducer3d_models import Inducer3DInputs, Inducer3DState


class PumpForge3DMainWindow:
    def __init__(self) -> None:
        self._dialog: QDialog = loadUi(self._ui_path())
        self.state = self._init_state_defaults()

    @staticmethod
    def _ui_path() -> str:
        return str(Path(__file__).resolve().parent.parent / "ui" / "PumpForge3D-Inducer.ui")

    @staticmethod
    def _init_state_defaults() -> Inducer3DState:
        return Inducer3DState(inputs=Inducer3DInputs())

    def _placeholder_future_methods(self) -> None:
        return None

    def show(self) -> None:
        self._dialog.show()

    def close(self) -> None:
        self._dialog.close()

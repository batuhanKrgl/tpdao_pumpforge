from pathlib import Path

from PyQt5.QtWidgets import QApplication

from apps.PumpForge3D.gui.main_window import PumpForge3DMainWindow


def test_pumpforge3d_main_window_imports() -> None:
    app = QApplication.instance() or QApplication([])
    window = PumpForge3DMainWindow()
    assert Path(window._ui_path()).exists()
    window.close()
    app.quit()

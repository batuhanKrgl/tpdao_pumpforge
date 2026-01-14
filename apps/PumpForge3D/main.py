import sys

from PyQt5.QtWidgets import QApplication

from apps.PumpForge3D.gui.main_window import PumpForge3DMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = PumpForge3DMainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

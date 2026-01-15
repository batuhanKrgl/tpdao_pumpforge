from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from apps.PumpForge3D.meridional.section_model import MeridionalSection2D


class MeridionalPlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)

    def plot_section(self, section: MeridionalSection2D) -> None:
        self.axes.clear()
        self.axes.plot(section.hub_curve[:, 0], section.hub_curve[:, 1], label="Hub")
        self.axes.plot(section.tip_curve[:, 0], section.tip_curve[:, 1], label="Tip")
        self.axes.plot(
            section.leading_edge[:, 0],
            section.leading_edge[:, 1],
            label="Leading Edge",
        )
        self.axes.plot(
            section.trailing_edge[:, 0],
            section.trailing_edge[:, 1],
            label="Trailing Edge",
        )

        self.axes.plot(
            section.hub_ctrl[:, 0],
            section.hub_ctrl[:, 1],
            linestyle="--",
            color="tab:blue",
            alpha=0.4,
        )
        self.axes.plot(
            section.tip_ctrl[:, 0],
            section.tip_ctrl[:, 1],
            linestyle="--",
            color="tab:orange",
            alpha=0.4,
        )
        self.axes.scatter(section.hub_ctrl[:, 0], section.hub_ctrl[:, 1], color="tab:blue")
        self.axes.scatter(section.tip_ctrl[:, 0], section.tip_ctrl[:, 1], color="tab:orange")
        self.axes.scatter(section.le_ctrl[:, 0], section.le_ctrl[:, 1], color="tab:green")
        self.axes.scatter(section.te_ctrl[:, 0], section.te_ctrl[:, 1], color="tab:red")

        self.axes.set_xlabel("z [m]")
        self.axes.set_ylabel("r [m]")
        self.axes.set_aspect("equal", adjustable="datalim")
        self.axes.legend(loc="best")
        self.axes.grid(True, linestyle=":", alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

from __future__ import annotations

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from apps.PumpForge3D.meridional.section_model import MeridionalSection2D


class MeridionalPlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)

    def plot_section(self, section: MeridionalSection2D) -> None:
        scale = 1000.0
        hub_curve = section.hub_curve * scale
        tip_curve = section.tip_curve * scale
        leading_edge = section.leading_edge * scale
        trailing_edge = section.trailing_edge * scale
        hub_ctrl = section.hub_ctrl * scale
        tip_ctrl = section.tip_ctrl * scale
        le_ctrl = section.le_ctrl * scale
        te_ctrl = section.te_ctrl * scale

        self.axes.clear()
        self.axes.plot(hub_curve[:, 0], hub_curve[:, 1], label="Hub")
        self.axes.plot(tip_curve[:, 0], tip_curve[:, 1], label="Tip")
        self.axes.plot(
            leading_edge[:, 0],
            leading_edge[:, 1],
            label="Leading Edge",
        )
        self.axes.plot(
            trailing_edge[:, 0],
            trailing_edge[:, 1],
            label="Trailing Edge",
        )

        self.axes.plot(
            hub_ctrl[:, 0],
            hub_ctrl[:, 1],
            linestyle="--",
            color="tab:blue",
            alpha=0.4,
        )
        self.axes.plot(
            tip_ctrl[:, 0],
            tip_ctrl[:, 1],
            linestyle="--",
            color="tab:orange",
            alpha=0.4,
        )
        self.axes.scatter(hub_ctrl[:, 0], hub_ctrl[:, 1], color="tab:blue")
        self.axes.scatter(tip_ctrl[:, 0], tip_ctrl[:, 1], color="tab:orange")
        self.axes.scatter(le_ctrl[:, 0], le_ctrl[:, 1], color="tab:green")
        self.axes.scatter(te_ctrl[:, 0], te_ctrl[:, 1], color="tab:red")

        z_min = min(hub_curve[:, 0].min(), tip_curve[:, 0].min())
        z_max = max(hub_curve[:, 0].max(), tip_curve[:, 0].max())
        self.axes.plot([z_min, z_max], [0.0, 0.0], color="black", linewidth=1.0)

        inlet_mid = 0.5 * (leading_edge[0] + leading_edge[-1])
        outlet_mid = 0.5 * (trailing_edge[0] + trailing_edge[-1])
        arrow_len = max(1.0, 0.1 * (z_max - z_min))
        self.axes.annotate(
            "",
            xy=(inlet_mid[0] + arrow_len, inlet_mid[1]),
            xytext=(inlet_mid[0], inlet_mid[1]),
            arrowprops={"arrowstyle": "->", "color": "tab:green"},
        )
        self.axes.annotate(
            "",
            xy=(outlet_mid[0] + arrow_len, outlet_mid[1]),
            xytext=(outlet_mid[0], outlet_mid[1]),
            arrowprops={"arrowstyle": "->", "color": "tab:red"},
        )

        self.axes.set_xlabel("z [mm]")
        self.axes.set_ylabel("r [mm]")
        self.axes.set_aspect(1.0, adjustable="datalim")
        self.axes.legend(loc="best")
        self.axes.grid(True, linestyle=":", alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

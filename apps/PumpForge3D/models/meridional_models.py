from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from apps.PumpForge3D.meridional.section_model import (
    MeridionalBezierParams,
    MeridionalSection2D,
)


@dataclass
class MeridionalSectionState:
    params: MeridionalBezierParams
    section: MeridionalSection2D
    section_serialized: Dict[str, List[List[float]]]


def serialize_section(section: MeridionalSection2D) -> Dict[str, List[List[float]]]:
    return {
        "hub_curve": section.hub_curve.tolist(),
        "tip_curve": section.tip_curve.tolist(),
        "leading_edge": section.leading_edge.tolist(),
        "trailing_edge": section.trailing_edge.tolist(),
        "hub_ctrl": section.hub_ctrl.tolist(),
        "tip_ctrl": section.tip_ctrl.tolist(),
        "le_ctrl": section.le_ctrl.tolist(),
        "te_ctrl": section.te_ctrl.tolist(),
    }

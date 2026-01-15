from __future__ import annotations

from math import comb

import numpy as np


def bernstein_poly(n: int, i: int, t: np.ndarray) -> np.ndarray:
    return comb(n, i) * np.power(t, i) * np.power(1 - t, n - i)


def bezier_curve(control_points: np.ndarray, t: np.ndarray) -> np.ndarray:
    if control_points.ndim != 2 or control_points.shape[1] != 2:
        raise ValueError("control_points must have shape (n_ctrl, 2)")
    n_ctrl = control_points.shape[0]
    if n_ctrl < 2:
        raise ValueError("control_points must contain at least two points")
    degree = n_ctrl - 1
    t = np.asarray(t)
    points = np.zeros((t.size, 2), dtype=float)
    for i in range(n_ctrl):
        points += bernstein_poly(degree, i, t)[:, None] * control_points[i]
    return points


def sample_bezier(control_points: np.ndarray, n_points: int = 200) -> np.ndarray:
    n_points = max(2, int(n_points))
    t = np.linspace(0.0, 1.0, n_points)
    return bezier_curve(control_points, t)

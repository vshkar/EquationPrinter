"""Evaluate SymPy expressions over a 2D grid and build Plotly figures."""

from typing import Optional

import numpy as np
import plotly.graph_objects as go
import sympy


def evaluate_function(
    expr: sympy.Expr,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n: int = 50,
) -> dict[str, np.ndarray]:
    """Evaluate a SymPy expression over a 2D uniform grid.

    Args:
        expr: SymPy expression with free symbols x, y.
        x_range: (min, max) for the x-axis.
        y_range: (min, max) for the y-axis.
        n: Number of sample points along each axis (n×n grid).

    Returns:
        Dict with keys ``X``, ``Y``, ``Z`` — each a 2D float64 array.
        Invalid positions (inf / nan) are replaced with ``np.nan`` so
        Plotly skips them without crashing.
    """
    x_vals = np.linspace(x_range[0], x_range[1], n)
    y_vals = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(x_vals, y_vals)

    f = sympy.lambdify(("x", "y"), expr, modules="numpy")

    with np.errstate(all="ignore"):
        Z = np.asarray(f(X, Y), dtype=np.float64)
        Z[np.isinf(Z) | np.isnan(Z)] = np.nan

    return {"X": X, "Y": Y, "Z": Z}


def create_plots(
    expr: sympy.Expr,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n: int = 50,
) -> dict[str, go.Figure]:
    """Build 3D-surface and 2D-contour Plotly figures for an expression.

    Args:
        expr: Validated SymPy expression.
        x_range: (min, max) for x-axis.
        y_range: (min, max) for y-axis.
        n: Grid resolution.

    Returns:
        ``{"surface": go.Figure, "contour": go.Figure}``.
    """
    data = evaluate_function(expr, x_range, y_range, n)
    X, Y, Z = data["X"], data["Y"], data["Z"]
    x_vals = X[0, :]
    y_vals = Y[:, 0]

    # --- 3D Surface ---
    surface = go.Figure()
    surface.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="viridis",
            colorbar=dict(title="z"),
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )
    surface.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode="auto",
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # --- 2D Contour ---
    contour = go.Figure()
    contour.add_trace(
        go.Contour(
            x=x_vals,
            y=y_vals,
            z=Z,
            colorscale="viridis",
            ncontours=20,
            contours=dict(coloring="heatmap"),
            colorbar=dict(title="z"),
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )
    contour.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return {"surface": surface, "contour": contour}

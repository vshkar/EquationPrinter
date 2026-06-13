"""3D Expression Plotter — Streamlit app entry point.

Type a mathematical expression (using ``x`` and ``y`` as variables) and
see an interactive 3D surface plot alongside a 2D contour plot.
"""

import numpy as np
import streamlit as st

from parser import parse_expression
from plotter import create_plots, evaluate_function
from stl_export import stl_to_bytes, surface_to_stl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXAMPLES: list[tuple[str, str]] = [
    ("sin(x)·cos(y)", "sin(x)*cos(y)"),
    ("x² + y²", "x^2 + y^2"),
    ("sin(x)cos(y)", "sin(x)cos(y)"),
    ("sin(x·y)", "sin(x*y)"),
    ("sin(√(x²+y²))", "sin(sqrt(x^2 + y^2))"),
    ("sinx + cosy", "sinx + cosy"),
    ("cos(x)·e^-(x²+y²)/10", "cos(x)*exp(-(x^2+y^2)/10)"),
    ("x³ - 3x·y²", "x^3 - 3*x*y^2"),
]

DEFAULT_EXPRESSION = "sin(x)*cos(y)"

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="3D Expression Plotter",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 3D Expression Plotter")

    # ---- Input row ----
    col_input, col_examples = st.columns([3, 1])

    with col_input:
        expr_text = st.text_input(
            "Enter a mathematical expression (use **x**, **y** as variables):",
            value=DEFAULT_EXPRESSION,
            key="expr_input",
            placeholder="e.g. sin(x)*cos(y)  or  x^2 + y^2",
        )

    with col_examples:
        st.caption("Examples")
        for label, value in EXAMPLES:
            if st.button(label, key=f"ex_{value}", use_container_width=True):
                st.session_state.expr_input = value
                st.rerun()

    # ---- Sidebar: range controls ----
    with st.sidebar:
        st.header("🎛️ Ranges")

        col_a, col_b = st.columns(2)
        with col_a:
            x_min = st.number_input("x min", value=-5.0, step=0.5, key="x_min")
            y_min = st.number_input("y min", value=-5.0, step=0.5, key="y_min")
        with col_b:
            x_max = st.number_input("x max", value=5.0, step=0.5, key="x_max")
            y_max = st.number_input("y max", value=5.0, step=0.5, key="y_max")

        resolution = st.slider(
            "Resolution",
            min_value=20,
            max_value=100,
            value=50,
            step=10,
            key="resolution",
            help="Number of sample points along each axis. Higher = smoother but slower.",
        )

        st.divider()
        st.header("📐 STL Export")

        base_z_mode = st.radio(
            "Base Z",
            ["Auto (min Z)", "Manual"],
            key="base_z_mode",
            horizontal=True,
            help="Auto: places the base slightly below the lowest surface point. "
                 "Manual: specify a fixed Z value for the flat bottom.",
        )

        if base_z_mode == "Manual":
            base_z_value = st.number_input(
                "Base Z value",
                value=-5.0,
                step=0.5,
                key="base_z_manual",
                help="Z-coordinate of the flat bottom surface.",
            )
        else:
            base_z_value = None  # computed after evaluation

        scale_mm = st.number_input(
            "Scale XY (mm per unit)",
            value=10.0,
            step=1.0,
            min_value=0.1,
            key="scale_mm",
            help="Multiply X and Y math coordinates by this factor to get "
                 "millimeters. E.g. scale=10 turns a −5…+5 range into a "
                 "100 mm model.",
        )

        z_scale = st.slider(
            "Z exaggeration",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="z_scale",
            help="Multiplier for the Z axis only. >1 exaggerates height, "
                 "<1 flattens. E.g. 2.0 makes peaks twice as tall.",
        )

        st.divider()

        # Display the parsed expression nicely
        if expr_text.strip():
            expr, error = parse_expression(expr_text)
            if expr is not None:
                st.caption("Parsed expression:")
                st.latex(f"z = {sp.latex(expr)}")

    # ---- Pipeline ----
    if not expr_text.strip():
        st.info("Enter an expression above to see the plot.")
        return

    with st.spinner("Parsing expression…"):
        expr, error = parse_expression(expr_text)

    if error:
        st.error(error, icon="❌")
        return

    # Safety: if ranges are inverted, swap them
    x_rng = (min(x_min, x_max), max(x_min, x_max))
    y_rng = (min(y_min, y_max), max(y_min, y_max))

    with st.spinner("Generating plots…"):
        try:
            figures = create_plots(expr, x_range=x_rng, y_range=y_rng, n=resolution)
        except Exception as exc:
            st.error(f"Error generating plots: {exc}", icon="💥")
            return

    # ---- Display plots ----
    col_3d, col_2d = st.columns(2)

    with col_3d:
        st.subheader("3D Surface")
        st.caption("Drag to rotate • Scroll to zoom • Right-drag to pan")
        st.plotly_chart(figures["surface"], use_container_width=True)

    with col_2d:
        st.subheader("2D Contour")
        st.caption("Drag to pan • Scroll to zoom • Double-click to reset")
        st.plotly_chart(figures["contour"], use_container_width=True)

    # ---- STL Export ----
    st.divider()
    st.header("📐 3D Print Export")

    with st.spinner("Building STL mesh…"):
        data = evaluate_function(expr, x_range=x_rng, y_range=y_rng, n=resolution)
        X, Y, Z = data["X"], data["Y"], data["Z"]

        # Determine base_z in auto mode
        if base_z_value is None:
            z_min = float(np.nanmin(Z))
            z_max = float(np.nanmax(Z))
            if np.isnan(z_min) or np.isinf(z_min):
                bz = 0.0
            elif np.isclose(z_min, z_max):
                bz = z_min - 1.0
            else:
                bz = z_min - 0.1 * (z_max - z_min)
        else:
            bz = float(base_z_value)

        stl_ready = False
        stl_data = b""
        try:
            mesh = surface_to_stl(X, Y, Z, base_z=bz, scale=scale_mm, z_scale=z_scale)
            stl_data = stl_to_bytes(mesh, binary=True)
            stl_ready = True
        except Exception as exc:
            st.error(f"Error generating STL: {exc}", icon="💥")

    col_info, col_dl = st.columns([2, 1])
    with col_info:
        st.caption(
            f"Base Z: **{bz:.3f}**  |  "
            f"Scale: **{scale_mm:.1f}** mm/unit  |  "
            f"Grid: **{resolution}**×**{resolution}**"
        )
        z_range = z_max - z_min if not (
            np.isnan(z_min) or np.isinf(z_min)
        ) else 1.0
        st.caption(
            f"Approx. model size: "
            f"**{scale_mm * (x_rng[1] - x_rng[0]):.0f}** × "
            f"**{scale_mm * (y_rng[1] - y_rng[0]):.0f}** × "
            f"**{scale_mm * z_scale * (z_range if not np.isclose(z_range, 0) else 1.0):.0f}** mm"
        )

    with col_dl:
        st.download_button(
            label="📥 Download STL",
            data=stl_data,
            file_name="surface.stl",
            mime="application/slicer-stl",
            disabled=not stl_ready,
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sympy as sp  # noqa: N813 — used inside main() for LaTeX rendering
    main()

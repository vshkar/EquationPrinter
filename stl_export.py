"""STL mesh generation for 3D printing.

Converts a gridded surface ``z = f(x, y)`` into a watertight STL mesh
suitable for slicing and 3D printing.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from stl import Mesh, Mode


def surface_to_stl(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    base_z: float | None = None,
    scale: float = 1.0,
    z_scale: float = 1.0,
) -> Mesh:
    """Build a watertight STL mesh from a gridded surface.

    The mesh consists of:

    * **Top surface** — triangulated from the *(X, Y, Z)* grid.
    * **Bottom surface** — a flat base at ``base_z``.
    * **Four side walls** — connecting the top perimeter to the base.

    NaN values in *Z* are capped to ``base_z`` so the mesh is always
    closed (watertight).

    Args:
        X: 2D array of x-coordinates, shape ``(n, n)``.
        Y: 2D array of y-coordinates, shape ``(n, n)``.
        Z: 2D array of z-coordinates, shape ``(n, n)``.  May contain NaN.
        base_z: Z-coordinate for the flat bottom.  If ``None``, computed
            as ``min(Z) - 0.1 * (max(Z) - min(Z))`` (10% below the lowest
            valid point), falling back to ``0.0`` for degenerate surfaces.
        scale: Multiply X and Y coordinates by this factor (math units → mm).
        z_scale: Additional multiplier for Z only.  Use >1 to exaggerate
            height, <1 to flatten.  Final Z = Z * scale * z_scale.

    Returns:
        A ``stl.mesh.Mesh`` instance with computed face normals.
    """
    n = X.shape[0]
    if X.shape != (n, n) or Y.shape != (n, n) or Z.shape != (n, n):
        raise ValueError(
            f"X, Y, Z must be square 2D arrays; "
            f"got shapes {X.shape}, {Y.shape}, {Z.shape}"
        )

    # --- Determine base_z ---
    z_valid = Z[np.isfinite(Z)]
    if base_z is None:
        if z_valid.size == 0:
            base_z = 0.0
        else:
            z_min = float(np.nanmin(Z))
            z_max = float(np.nanmax(Z))
            if np.isclose(z_min, z_max):
                base_z = z_min - 1.0
            else:
                base_z = z_min - 0.1 * (z_max - z_min)
    base_z = float(base_z)

    # --- Cap NaN / inf values to base_z ---
    Z_clean = Z.copy()
    bad = np.isnan(Z_clean) | np.isinf(Z_clean)
    Z_clean[bad] = base_z

    # --- Flattened coordinates ---
    z_factor = scale * z_scale
    xf = (X * scale).ravel()
    yf = (Y * scale).ravel()
    zt = (Z_clean * z_factor).ravel()  # top Z
    zb = np.full_like(zt, base_z * z_factor)  # bottom Z

    # --- Allocate mesh data ---
    n_cells = (n - 1) * (n - 1)
    total_tris = 4 * n_cells + 8 * (n - 1)  # = 4n² − 4
    data = np.zeros(total_tris, dtype=Mesh.dtype)
    tidx = 0

    # --- Top surface ---
    for j in range(n - 1):
        for i in range(n - 1):
            bl = j * n + i
            br = j * n + i + 1
            tl = (j + 1) * n + i
            tr = (j + 1) * n + i + 1
            # Winding ccw viewed from +Z
            _tri(data, tidx, xf[tl], yf[tl], zt[tl],
                 xf[bl], yf[bl], zt[bl],
                 xf[tr], yf[tr], zt[tr])
            tidx += 1
            _tri(data, tidx, xf[tr], yf[tr], zt[tr],
                 xf[br], yf[br], zt[br],
                 xf[bl], yf[bl], zt[bl])
            tidx += 1

    # --- Bottom surface (reversed winding for −Z normal) ---
    for j in range(n - 1):
        for i in range(n - 1):
            bl = j * n + i
            br = j * n + i + 1
            tl = (j + 1) * n + i
            tr = (j + 1) * n + i + 1
            _tri(data, tidx, xf[bl], yf[bl], zb[bl],
                 xf[tr], yf[tr], zb[tr],
                 xf[tl], yf[tl], zb[tl])
            tidx += 1
            _tri(data, tidx, xf[bl], yf[bl], zb[bl],
                 xf[br], yf[br], zb[br],
                 xf[tr], yf[tr], zb[tr])
            tidx += 1

    # --- Side walls ---
    tidx = _add_bottom_wall(data, tidx, n, xf, yf, zt, zb)
    tidx = _add_right_wall(data, tidx, n, xf, yf, zt, zb)
    tidx = _add_top_wall(data, tidx, n, xf, yf, zt, zb)
    tidx = _add_left_wall(data, tidx, n, xf, yf, zt, zb)

    mesh = Mesh(data, remove_empty_areas=False, calculate_normals=False)
    mesh.update_normals()
    return mesh


def stl_to_bytes(stl_mesh: Mesh, binary: bool = True) -> bytes:
    """Serialize a Mesh to bytes for download.

    Args:
        stl_mesh: A ``stl.mesh.Mesh`` instance.
        binary: If ``True`` (default), produce binary STL; otherwise ASCII.

    Returns:
        STL file contents as bytes.
    """
    buf = BytesIO()
    stl_mesh.save("surface.stl", fh=buf, mode=Mode.BINARY if binary else Mode.ASCII)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tri(
    data: np.ndarray,
    idx: int,
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    x3: float, y3: float, z3: float,
) -> None:
    """Write one triangle's vertex positions to ``data["vectors"][idx]``."""
    data["vectors"][idx] = [
        [x1, y1, z1],
        [x2, y2, z2],
        [x3, y3, z3],
    ]


def _add_bottom_wall(
    data: np.ndarray, tidx: int, n: int,
    xf: np.ndarray, yf: np.ndarray,
    zt: np.ndarray, zb: np.ndarray,
) -> int:
    """Bottom edge: j = 0, i from 0 → n−2.  Normal points −Y."""
    j = 0
    for i in range(n - 1):
        a = j * n + i
        b = j * n + i + 1
        _wall_quad(data, tidx, a, b, xf, yf, zt, zb,
                   outward=(0, -1))  # bottom-left → bottom-right
        tidx += 2
    return tidx


def _add_right_wall(
    data: np.ndarray, tidx: int, n: int,
    xf: np.ndarray, yf: np.ndarray,
    zt: np.ndarray, zb: np.ndarray,
) -> int:
    """Right edge: i = n−1, j from 0 → n−2.  Normal points +X."""
    i = n - 1
    for j in range(n - 1):
        a = j * n + i
        b = (j + 1) * n + i
        _wall_quad(data, tidx, a, b, xf, yf, zt, zb,
                   outward=(1, 0))
        tidx += 2
    return tidx


def _add_top_wall(
    data: np.ndarray, tidx: int, n: int,
    xf: np.ndarray, yf: np.ndarray,
    zt: np.ndarray, zb: np.ndarray,
) -> int:
    """Top edge: j = n−1, i from n−2 → 0 (reversed).  Normal points +Y."""
    j = n - 1
    for i in range(n - 2, -1, -1):
        a = j * n + i + 1  # right → left for outward normal
        b = j * n + i
        _wall_quad(data, tidx, a, b, xf, yf, zt, zb,
                   outward=(0, 1))
        tidx += 2
    return tidx


def _add_left_wall(
    data: np.ndarray, tidx: int, n: int,
    xf: np.ndarray, yf: np.ndarray,
    zt: np.ndarray, zb: np.ndarray,
) -> int:
    """Left edge: i = 0, j from n−2 → 0 (reversed).  Normal points −X."""
    i = 0
    for j in range(n - 2, -1, -1):
        a = (j + 1) * n + i  # top → bottom for outward normal
        b = j * n + i
        _wall_quad(data, tidx, a, b, xf, yf, zt, zb,
                   outward=(-1, 0))
        tidx += 2
    return tidx


def _wall_quad(
    data: np.ndarray,
    tidx: int,
    idx_a: int,
    idx_b: int,
    xf: np.ndarray,
    yf: np.ndarray,
    zt: np.ndarray,
    zb: np.ndarray,
    outward: tuple[int, int],
) -> None:
    """Emit two triangles for one wall quad.

    Quad vertices in ccw order viewed from *outside*:
    ``q_a → q_b → p_b → p_a``, where *q* are bottom points and *p* are
    top points.  *outward* is a signed direction hint ``(dx, dy)`` used
    purely for readability — the winding is what actually determines the
    normal.
    """
    # Bottom vertices (q) share XY with top vertices (p) at the same index
    qa_x, qa_y, qa_z = xf[idx_a], yf[idx_a], zb[idx_a]
    qb_x, qb_y, qb_z = xf[idx_b], yf[idx_b], zb[idx_b]
    pa_x, pa_y, pa_z = xf[idx_a], yf[idx_a], zt[idx_a]
    pb_x, pb_y, pb_z = xf[idx_b], yf[idx_b], zt[idx_b]

    # Triangle 1: q_a → q_b → p_b
    _tri(data, tidx, qa_x, qa_y, qa_z,
         qb_x, qb_y, qb_z,
         pb_x, pb_y, pb_z)
    # Triangle 2: p_b → p_a → q_a
    _tri(data, tidx + 1, pb_x, pb_y, pb_z,
         pa_x, pa_y, pa_z,
         qa_x, qa_y, qa_z)

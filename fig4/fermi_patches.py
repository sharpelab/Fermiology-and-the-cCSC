"""
fermi_patches.py
----------------
Matplotlib patch helpers for common Fermi surface topologies.

All shapes use periodic cubic splines (scipy.interpolate.splprep).
The spline is always fitted in unit/normalised space (radius = 1, no aspect
correction) and then scaled to data coords, so the parametric fit is
always well-conditioned regardless of how extreme the axis aspect ratio is.

Aspect-ratio correction is applied as a final y-scale:
    scale_y = scale_x * ar,  ar = (px/data-unit in x) / (px/data-unit in y)
so shapes always appear visually circular / correct on screen.

Functions
---------
make_circular_patch      – simply-connected disk
make_annular_patch       – ring with circular hole
make_three_pocket_patch  – three tangentially-oriented ellipses (C3)
make_bean_pocket_patch   – smooth open-arc bean + small elliptical pocket
add_patches              – add a list of patches to an axis
refresh_patches          – rebuild patches after axis limits / figure resize
"""

import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

__all__ = [
    "make_circular_patch",
    "make_annular_patch",
    "make_three_pocket_patch",
    "make_bean_pocket_patch",
    "add_patches",
    "refresh_patches",
]

# ---------------------------------------------------------------------------
# Aspect-ratio helper
# ---------------------------------------------------------------------------

def _aspect_ratio(ax):
    """
    Return (px per x-data-unit) / (px per y-data-unit).

    Multiply the y scale-factor of any shape by this value so it appears
    visually correct regardless of axis limits or figure dimensions.
    Uses figure size + axes position fraction (available before rendering).
    """
    fig = ax.figure
    fw, fh = fig.get_size_inches()
    pos = ax.get_position()
    ax_w = fw * pos.width
    ax_h = fh * pos.height
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    if ax_h == 0 or dy == 0:
        return 1.0
    return (ax_w / dx) / (ax_h / dy)


# ---------------------------------------------------------------------------
# Core spline builder  –  always fits in unit space, then scales
# ---------------------------------------------------------------------------

def _spline_path(kx_unit, ky_unit, cx, cy, sx, sy, n_eval=600):
    """
    Fit a periodic cubic spline to unit key points, then map to data coords.

    Parameters
    ----------
    kx_unit, ky_unit : key-point coordinates in unit space (O(1) magnitude)
    cx, cy           : centre in data coordinates
    sx, sy           : x- and y-scale factors (data units per unit radius)
                       set sy = sx * ar to get visually correct shapes
    n_eval           : number of output path points
    """
    tck, _ = splprep([kx_unit, ky_unit], s=0, per=True, k=3)
    t = np.linspace(0, 1, n_eval, endpoint=False)
    xs, ys = splev(t, tck)
    fx = cx + xs * sx
    fy = cy + ys * sy
    verts = list(zip(fx, fy)) + [(fx[0], fy[0])]
    codes = [Path.MOVETO] + [Path.LINETO] * (n_eval - 1) + [Path.CLOSEPOLY]
    return Path(verts, codes)


# ---------------------------------------------------------------------------
# Unit-space shape generators  (all O(1), centred at origin, NO aspect fix)
# ---------------------------------------------------------------------------

def _circle_unit(n=10, cw=False):
    """N points on the unit circle (CCW by default, CW if cw=True)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if cw:
        theta = theta[::-1]
    return np.cos(theta), np.sin(theta)


def _ellipse_unit(minor_frac, angle_deg, n=10):
    """
    Unit ellipse: semi-major = 1 along the rotated x-axis,
    semi-minor = minor_frac along the rotated y-axis.
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    phi = np.radians(angle_deg)
    ex = np.cos(theta) * np.cos(phi) - minor_frac * np.sin(theta) * np.sin(phi)
    ey = np.cos(theta) * np.sin(phi) + minor_frac * np.sin(theta) * np.cos(phi)
    return ex, ey


def _pacman_unit(gap_angle=50, n_outer=9):
    """
    Unit Pac-Man (radius = 1, mouth on the +x side).
    Outer arc from +gap_angle → 360-gap_angle going CCW through 180°,
    then two inner lip tips for a smooth closed mouth.
    """
    g = np.radians(gap_angle)
    outer_a = np.linspace(g, 2 * np.pi - g, n_outer)
    ox, oy = np.cos(outer_a), np.sin(outer_a)
    ir, iy = -0.3,0.1          # inward reach and y-spread of lip tips
    return (np.concatenate([ox, [ir,  ir]]),
            np.concatenate([oy, [-iy, iy]]))


# ---------------------------------------------------------------------------
# Linewidth alias normalisation  (lw= vs linewidth=)
# ---------------------------------------------------------------------------

def _resolve_lw(linewidth, kwargs):
    """Pop 'lw' from kwargs if present; it wins over the default linewidth."""
    if 'lw' in kwargs:
        linewidth = kwargs.pop('lw')
    return linewidth, kwargs


# ---------------------------------------------------------------------------
# Public patch constructors
# ---------------------------------------------------------------------------

def make_circular_patch(
    ax,
    center=(0, 0),
    radius=1.0,
    facecolor="#4daf4a",
    edgecolor="black",
    linewidth=3,
    n_key=10,
    **kwargs,
):
    """Simply-connected circular Fermi pocket."""
    linewidth, kwargs = _resolve_lw(linewidth, kwargs)
    ar = _aspect_ratio(ax)
    kx, ky = _circle_unit(n=n_key)
    cx, cy = center
    path = _spline_path(kx, ky, cx, cy, radius, radius * ar)
    return [PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                      linewidth=linewidth, **kwargs)]


def make_annular_patch(
    ax,
    center=(0, 0),
    outer_radius=1.0,
    inner_radius=0.45,
    facecolor="#5b8db8",
    edgecolor="black",
    outer_edgecolor=None,
    inner_edgecolor=None,
    linewidth=3,
    outer_linewidth=None,
    inner_linewidth=None,
    n_key=10,
    **kwargs,
):
    """
    Annular (ring-shaped) Fermi surface.

    Edge colors may be set jointly via ``edgecolor`` or overridden with
    ``outer_edgecolor`` / ``inner_edgecolor``.  Returns two patches:
      [0] compound fill patch  – ring fill + outer edge
      [1] inner edge overlay   – redraws inner contour in inner_edgecolor
    """
    linewidth, kwargs = _resolve_lw(linewidth, kwargs)
    ec_out = outer_edgecolor if outer_edgecolor is not None else edgecolor
    ec_in  = inner_edgecolor if inner_edgecolor is not None else edgecolor
    lw_out = outer_linewidth if outer_linewidth is not None else linewidth
    lw_in  = inner_linewidth if inner_linewidth is not None else linewidth

    ar = _aspect_ratio(ax)
    cx, cy = center
    kx_ccw, ky_ccw = _circle_unit(n=n_key, cw=False)
    kx_cw,  ky_cw  = _circle_unit(n=n_key, cw=True)

    op = _spline_path(kx_ccw, ky_ccw, cx, cy, outer_radius, outer_radius * ar)
    ip = _spline_path(kx_cw,  ky_cw,  cx, cy, inner_radius, inner_radius * ar)

    # Compound path: fills ring + draws outer edge
    verts = np.vstack([op.vertices, ip.vertices])
    codes = np.concatenate([op.codes, ip.codes])
    fill = PathPatch(Path(verts, codes), facecolor=facecolor,
                     edgecolor=ec_out, linewidth=lw_out, **kwargs)

    # Inner edge overlay
    base_z  = kwargs.get("zorder", 1)
    inner_kw = {k: v for k, v in kwargs.items() if k != "zorder"}
    ip2 = _spline_path(kx_cw, ky_cw, cx, cy, inner_radius, inner_radius * ar)
    overlay = PathPatch(ip2, facecolor="none", edgecolor=ec_in,
                        linewidth=lw_in, zorder=base_z + 0.1, **inner_kw)
    return [fill, overlay]


def make_three_pocket_patch(
    ax,
    center=(0, 0),
    orbit_radius=0.65,
    pocket_major=0.55,
    pocket_minor=0.18,
    start_angle=90,
    facecolor="#e41a1c",
    edgecolor="black",
    linewidth=3,
    n_key=10,
    **kwargs,
):
    """
    Three elliptical pockets at 120° intervals, tangentially oriented.

    Parameters
    ----------
    orbit_radius  : distance from center to pocket centres (x data units)
    pocket_major  : full tangential (long) axis length
    pocket_minor  : full radial (short) axis length
    start_angle   : angle (°) of the first pocket
    """
    linewidth, kwargs = _resolve_lw(linewidth, kwargs)
    ar = _aspect_ratio(ax)
    minor_frac = pocket_minor / pocket_major   # < 1
    patches = []
    for i in range(3):
        pos_deg = start_angle + i * 120
        pos_rad = np.radians(pos_deg)
        px = center[0] + orbit_radius * np.cos(pos_rad)
        py = center[1] + orbit_radius * np.sin(pos_rad) * ar   # ar-correct position
        kx, ky = _ellipse_unit(minor_frac, pos_deg + 90, n=n_key)
        path = _spline_path(kx, ky, px, py,
                             pocket_major / 2, pocket_major / 2 * ar)
        patches.append(PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                                 linewidth=linewidth, **kwargs))
    return patches


def make_bean_pocket_patch(
    ax,
    center=(0, 0),
    bean_radius=0.90,
    gap_angle=50,
    pocket_major=0.55,
    pocket_minor=0.22,
    pocket_offset=(1.55, 0),
    bean_facecolor="#ff7f00",
    pocket_facecolor=None,
    edgecolor="black",
    bean_edgecolor=None,
    pocket_edgecolor=None,
    linewidth=3,
    bean_linewidth=None,
    pocket_linewidth=None,
    n_key=9,
    **kwargs,
):
    """
    Smooth open-arc (Pac-Man) bean + small elliptical pocket.

    Parameters
    ----------
    center         : (x, y) geometric centre of the bean arc
    bean_radius    : arc radius in x data units
    gap_angle      : half-angle of mouth opening in degrees
    pocket_major   : full height (y axis) of the small ellipse
    pocket_minor   : full width  (x axis) of the small ellipse
    pocket_offset  : (dx, dy) of pocket centre relative to ``center``
    edgecolor      : fallback edge color for both shapes
    bean_edgecolor / pocket_edgecolor : per-shape overrides
    linewidth      : fallback line width; use ``lw=`` as alias
    bean_linewidth / pocket_linewidth : per-shape overrides
    """
    linewidth, kwargs = _resolve_lw(linewidth, kwargs)
    if pocket_facecolor is None:
        pocket_facecolor = bean_facecolor
    ec_bean   = bean_edgecolor   if bean_edgecolor   is not None else edgecolor
    ec_pocket = pocket_edgecolor if pocket_edgecolor is not None else edgecolor
    lw_bean   = bean_linewidth   if bean_linewidth   is not None else linewidth
    lw_pocket = pocket_linewidth if pocket_linewidth is not None else linewidth

    ar = _aspect_ratio(ax)
    cx, cy = center

    # Bean: unit pac-man scaled by bean_radius in x and bean_radius*ar in y
    kx, ky = _pacman_unit(gap_angle, n_outer=n_key)
    bean = PathPatch(_spline_path(kx, ky, cx, cy, bean_radius, bean_radius * ar),
                     facecolor=bean_facecolor, edgecolor=ec_bean,
                     linewidth=lw_bean, **kwargs)

    # Pocket: unit circle scaled by (pocket_minor/2, pocket_major/2 * ar)
    #         → visually correct vertical ellipse
    px = cx + pocket_offset[0]
    py = cy + pocket_offset[1]
    kx_e, ky_e = _circle_unit(n=10)
    pocket = PathPatch(
        _spline_path(kx_e, ky_e, px, py, pocket_minor / 2, pocket_major / 2 * ar),
        facecolor=pocket_facecolor, edgecolor=ec_pocket,
        linewidth=lw_pocket, **kwargs)
    return [bean, pocket]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def add_patches(ax, patches):
    """Add a list of patches to *ax*."""
    for p in patches:
        ax.add_patch(p)


def refresh_patches(ax, patches, make_fn, **kwargs):
    """
    Remove *patches* from *ax*, recreate via ``make_fn(ax, **kwargs)``,
    re-add, and return the new list.

    Call after changing axis limits or resizing the figure.
    """
    for p in patches:
        p.remove()
    new = make_fn(ax, **kwargs)
    add_patches(ax, new)
    return new


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))

    specs = [
        ("Simply connected",
         make_circular_patch,
         dict(center=(0, 0), radius=0.8, facecolor="#4daf4a", linewidth=3)),
        ("Annular",
         make_annular_patch,
         dict(center=(0, 0), outer_radius=0.8, inner_radius=0.35,
              facecolor="#5b8db8",
              outer_edgecolor="black", inner_edgecolor="crimson",
              linewidth=3)),
        ("Three pocket",
         make_three_pocket_patch,
         dict(center=(0, 0), orbit_radius=0.52, pocket_major=0.44,
              pocket_minor=0.15, start_angle=90,
              facecolor="#e41a1c", linewidth=3)),
        ("Bean + pocket",
         make_bean_pocket_patch,
         dict(center=(-0.4, 0), bean_radius=0.72, gap_angle=50,
              pocket_major=0.48, pocket_minor=0.19, pocket_offset=(1.28, 0),
              bean_facecolor="#984ea3",
              bean_edgecolor="black", pocket_edgecolor="darkorange",
              linewidth=3)),
    ]

    for ax, (title, fn, kw) in zip(axes, specs):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.0, 1.0)
        patches = fn(ax, **kw)
        add_patches(ax, patches)
        ax.set_aspect("auto")   # deliberately non-equal
        ax.axis("off")
        ax.set_title(title, fontsize=8)

    fig.tight_layout(pad=0.4)
    plt.savefig("/mnt/user-data/outputs/fermi_patches_demo.png",
                dpi=180, bbox_inches="tight")
    print("done")
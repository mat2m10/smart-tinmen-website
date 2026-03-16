import numpy as np
from matplotlib.patches import Circle

# ── Reference canvas size (your "good" render: 6" × 140 dpi = 840 px) ──
_REF_FIGSIZE_INCHES = 6.0
_REF_DPI = 140
_REF_PX = _REF_FIGSIZE_INCHES * _REF_DPI  # 840


def _px_to_data_dy(ax, px):
    """
    Convert a vertical distance from 'reference pixels' to data-space units.

    At the reference size (6in × 140dpi) this returns exactly the same value
    as the original.  At any other figsize/dpi it scales the input so that
    e.g. "30 px" always means the same *fraction* of the canvas.
    """
    fig = ax.figure
    current_px = fig.get_figheight() * fig.dpi
    s = current_px / _REF_PX

    inv = ax.transData.inverted()
    _, y0 = inv.transform((0.0, 0.0))
    _, y1 = inv.transform((0.0, float(px) * s))
    return y1 - y0


def _px_to_data_dx(ax, px):
    """Horizontal counterpart of _px_to_data_dy."""
    fig = ax.figure
    current_px = fig.get_figwidth() * fig.dpi
    s = current_px / _REF_PX

    inv = ax.transData.inverted()
    x0, _ = inv.transform((0.0, 0.0))
    x1, _ = inv.transform((float(px) * s, 0.0))
    return x1 - x0


def background_circle(ax,
                      center=(0.5, 0.5),
                      radius=0.75,
                      color="#EEF2FF",
                      alpha=1.0,
                      edgecolor="none",
                      lw=0.0,
                      zorder=0):
    """
    Draw a simple background circle behind everything.
    """
    cx, cy = map(float, center)
    r = float(radius)

    bg = Circle(
        (cx, cy),
        radius=r,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=float(lw),
        alpha=float(alpha),
        zorder=int(zorder),
    )
    ax.add_patch(bg)
    return bg
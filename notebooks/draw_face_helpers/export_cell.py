# ============================================================
# EXPORT CELL — paste this into draw_face.ipynb as a new cell
# (replaces or sits below your existing demo() cell)
# ============================================================

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from draw_face_helpers import (
    draw_face_patch, draw_face_brain, add_circle_lens_shadow,
    add_one_brain_circuit_from_brain_top,
    add_rect_bent_double_node_from_brain_top,
    draw_eyes, add_smile, background_circle,
)

# ── Reference size (must match geometry.py's _REF_PX) ────────────────
REF_PX = 840          # 6" × 140 dpi

# ── Export targets ────────────────────────────────────────────────────
TARGETS = {
    "favicon-16":           16,
    "favicon-32":           32,
    "favicon-48":           48,
    "logo-64":              64,
    "logo-128":             128,
    "apple-touch-icon-180": 180,
    "android-192":          192,
    "logo-256":             256,
    "logo-512":             512,
    "android-512":          512,
    "logo-1024":           1024,
}
ICO_SIZES = [16, 32, 48]
OUT_DIR = "logo_output"


def render_logo(target_px, simplify_below=64):
    """
    Render the logo at target_px × target_px.

    DPI stays at 140 (reference) so all _px params inside helpers
    produce correct data-space offsets via the patched geometry.py.
    figsize shrinks to hit the pixel target.

    Only point-based params (lw, wire_lw, etc.) are scaled by
    target_px / REF_PX.
    """
    scale = target_px / REF_PX
    is_tiny = target_px < simplify_below

    # ── figsize varies, DPI stays at 140 ─────────────────────────
    dpi = DPI
    figsize = target_px / dpi

    # ── Scale point-based stroke widths ──────────────────────────
    lw       = LW_OUTLINE * scale
    bg_lw    = 6 * scale
    wire_lw  = lw * 0.50
    wink_lw  = lw * 0.75
    smile_lw = lw * 0.80

    # Floors for tiny sizes
    if is_tiny:
        lw       = max(lw, 1.8)
        bg_lw    = max(bg_lw, 1.0)
        wire_lw  = max(wire_lw, 1.0)
        wink_lw  = max(wink_lw, 1.2)
        smile_lw = max(smile_lw, 1.2)

    # ── Figure ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()

    # Face outline
    fo = draw_face_patch(
        width=FACE_SIZE, height=FACE_SIZE, angle_top_to_bottom=0,
        rounding_top=FACE_ROUND_TOP, rounding_bottom=FACE_ROUND_BOT,
        cx=0.5, cy=0.5,
        facecolor="none", edgecolor=LINE_COLOR, lw=lw, zorder=20,
    )
    ax.add_patch(fo)

    # Brain / face fills (all _px params unchanged — geometry.py handles scaling)
    draw_face_brain(
        ax, fo, cx=0.5, cy=0.5,
        y_pos_px=30, curvature_px=-55, curve_dir="down",
        fill_pad_px=0, sep_n=700, boundary_interp=220,
        lw=lw,
        brain_color=BRAIN_COLOR, face_color=FACE_COLOR, line_color=LINE_COLOR,
    )

    # Lens shadow — skip for tiny
    if not is_tiny:
        add_lens_shadow_subdivisions_with_colors(
            ax, fo,
            center_start=(0.5 - 1.22, 0.5),
            center_end=(0.5 - 1.20, 0.5),
            radius=1.52,
            face_colors=face_band_colors,
            brain_colors=brain_band_colors,
            alpha=0.035, zorder=SHADOW_Z,
        )

    # Circuits — skip for tiny
    if not is_tiny:
        for shift in [-0.02, -0.30]:
            add_one_brain_circuit_from_brain_top(
                ax, fo, center_shift=shift, length_px=80,
                node_r=0.018, node_facecolor="#FFFFFF", node_edgecolor=NODE_EDGE,
                wire_lw=wire_lw, wire_color=WIRE_COLOR, zorder=CIRCUIT_Z,
            )
        for shift, bdir in [(-0.02, -1), (0.20, +1)]:
            add_rect_bent_double_node_from_brain_top(
                ax, fo, center_shift=shift, length_px=75, bend_px=40,
                bend_at=0.7, bend_dir=bdir, corner_px=12, corner_n=7,
                node1_on="vertical1", node1_u=0.58,
                node_facecolor1=NODE_LED, node_facecolor2=NODE_METAL,
                node_edgecolor=NODE_EDGE, node_r=0.018,
                wire_color=WIRE_COLOR, wire_lw=wire_lw, zorder=CIRCUIT_Z,
            )

    # Eyes
    draw_eyes(
        ax, fo, cx=0.5, cy=EYES_CY,
        open_left=False, open_right=True,
        eye_dx=0.19, eye_dy=-0.02, r=0.066,
        wink_lw=wink_lw, outline_color=LINE_COLOR,
        iris_color_left=EYE_IRIS, iris_color_right=EYE_IRIS,
        ring_ratio=0.10, iris_ratio=0.38,
        pupil_ratio=EYE_PUPIL_RATIO, spec_ratio=EYE_SPEC_RATIO,
        spec_offset=(0.30, 0.34), iris_offset_y=0.0,
        gaze_left=(1.0, 0.0), gaze_right=(-1.0, 0.0),
        gaze_strength=EYE_GAZE_STRENGTH, aa_shrink=0.995,
    )

    # Smile
    add_smile(
        ax, fo, cx=0.5, cy=0.5, y=SMILE_Y,
        w=SMILE_W, h=SMILE_H, theta1=210, theta2=330,
        lw=smile_lw, color=LINE_COLOR,
    )

    # Background circle
    background_circle(
        ax, center=(0.5, 0.5), radius=0.60,
        color=BACKGROUND_COLOR, alpha=0.90,
        edgecolor="#0B0F14", lw=bg_lw, zorder=0,
    )

    # Redraw outline on top
    top = draw_face_patch(
        width=FACE_SIZE, height=FACE_SIZE, angle_top_to_bottom=0,
        rounding_top=FACE_ROUND_TOP, rounding_bottom=FACE_ROUND_BOT,
        cx=0.5, cy=0.5,
        facecolor="none", edgecolor=LINE_COLOR, lw=lw, zorder=1000,
    )
    ax.add_patch(top)
    ax.relim()
    ax.autoscale_view()
    ax.margins(0.06)

    return fig


def fig_to_pil(fig, target_px):
    """Render figure to PIL Image at exact target dimensions."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0.02, transparent=False)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    if img.size != (target_px, target_px):
        img = img.resize((target_px, target_px), Image.LANCZOS)
    return img


# ── Run the export ───────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
images = {}

for name, px in sorted(TARGETS.items(), key=lambda x: x[1]):
    print(f"  {name} ({px}×{px}) …")
    fig = render_logo(px)
    img = fig_to_pil(fig, px)
    img.save(os.path.join(OUT_DIR, f"{name}.png"))
    images[px] = img

# Build favicon.ico
ico_imgs = [images[s] for s in ICO_SIZES if s in images]
if ico_imgs:
    ico_imgs[0].save(
        os.path.join(OUT_DIR, "favicon.ico"),
        format="ICO",
        sizes=[(s, s) for s in ICO_SIZES],
        append_images=ico_imgs[1:],
    )
    print(f"  favicon.ico ({', '.join(map(str, ICO_SIZES))} px)")

print(f"\nDone → {OUT_DIR}/")

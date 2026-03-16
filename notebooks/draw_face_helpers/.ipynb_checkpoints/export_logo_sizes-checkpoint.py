"""
Export the logo at multiple sizes with proper scaling.

THE PROBLEM:
  Your helpers already accept lw, wire_lw, node_r, etc. as arguments.
  The issue is that demo() passes FIXED values (LW_OUTLINE=11) at every
  size. In matplotlib, lw is in points (absolute) — it doesn't shrink
  when the canvas shrinks.

THE FIX:
  This script keeps figsize=6" constant and varies DPI to hit the target
  pixel count. That means data-space coordinates (node_r, eye_r, positions)
  stay unchanged. Only point-based params (lw, wire_lw, etc.) get scaled:
      scale = target_px / 840

  For tiny sizes (<64px), circuits & lens shadows are dropped — they're
  unreadable noise at favicon scale.

NO CHANGES to draw_face_helpers needed.

USAGE:
  Place next to your draw_face_helpers package, then:
      python export_logo_sizes.py
  Output: ./logo_output/
"""

import os, sys, colorsys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from draw_face_helpers import (
    draw_face_patch, draw_face_brain, add_circle_lens_shadow,
    add_one_brain_circuit_from_brain_top,
    add_rect_bent_double_node_from_brain_top,
    draw_eyes, add_smile, background_circle,
)

# ═══════════════════════════════════════════════════════════
#  Reference: your current "good" render (6" × 140dpi = 840px)
# ═══════════════════════════════════════════════════════════
REF_PX         = 840
REF_LW_OUTLINE = 11
REF_BG_LW      = 6

# Data-space (constant because figsize stays 6")
FACE_SIZE, FACE_ROUND_TOP, FACE_ROUND_BOT = 0.86, 0.44, 0.44
NODE_R, EYE_R = 0.018, 0.066

# Colors
BRAIN_COLOR, FACE_COLOR, LINE_COLOR = "#EEF2F8", "#FFFFFF", "#0B0F14"
BACKGROUND_COLOR = "#7FE6D4"
NODE_LED, NODE_METAL, NODE_EDGE = "#26D3B3", "#334155", LINE_COLOR
WIRE_COLOR, EYE_IRIS = LINE_COLOR, "#26D3B3"
EYE_PUPIL_RATIO, EYE_SPEC_RATIO, EYE_GAZE_STRENGTH = 0.35, 0.22, 0.45
EYES_CY, SMILE_Y, SMILE_W, SMILE_H = 0.42, -0.14, 0.36, 0.20

# Shadow band colors (from your original)
def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))
def _rgb01_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c*255)) for c in rgb))
def derive_cold_tint(bh, hs=-0.06, sf=0.35, vf=0.75, vs=0.92):
    r,g,b = _hex_to_rgb01(bh); h,s,v = colorsys.rgb_to_hsv(r,g,b)
    return _rgb01_to_hex(colorsys.hsv_to_rgb((h+hs)%1, max(s,sf), max(v*vs,vf)))
def cold_shadow_palette(bh, n=1, smin=0.08, smax=0.14, vmin=0.97, vmax=0.90):
    r,g,b = _hex_to_rgb01(bh); h,s,v = colorsys.rgb_to_hsv(r,g,b)
    return [_rgb01_to_hex(colorsys.hsv_to_rgb(h,ss,vv))
            for ss,vv in zip(np.linspace(smin,smax,n), np.linspace(vmin,vmax,n))]

CT_FACE  = derive_cold_tint(FACE_COLOR,  hs=-0.07, sf=0.30, vf=0.72, vs=0.90)
CT_BRAIN = derive_cold_tint(BRAIN_COLOR, hs=+0.10, sf=0.45, vf=0.68, vs=0.88)
FACE_BAND  = cold_shadow_palette(CT_FACE,  n=1)
BRAIN_BAND = cold_shadow_palette(CT_BRAIN, n=1, smin=0.10, smax=0.18, vmin=0.96, vmax=0.86)


# ═══════════════════════════════════════════════════════════
#  Scalable render
# ═══════════════════════════════════════════════════════════
def render_logo(target_px, transparent=False, simplify_below=64):
    scale = target_px / REF_PX

    # STRATEGY: Keep DPI=140 (same as reference) so that _px_to_data_dy/dx
    # inside your helpers produce the same data-space offsets as the 840px
    # render. We vary figsize to hit the target pixel count.
    # This means y_pos_px=30, curvature_px=-55, length_px=80 etc. all
    # "just work" without needing to scale them.
    dpi = 140
    figsize = target_px / dpi   # e.g. 16/140 = 0.114 inches

    # Scale all point-based params (lw is in points = 1/72 inch, absolute)
    lw   = REF_LW_OUTLINE * scale
    bglw = REF_BG_LW * scale
    is_tiny = target_px < simplify_below

    # Floor for tiny sizes so strokes don't vanish
    if is_tiny:
        lw   = max(lw, 1.8)
        bglw = max(bglw, 1.0)

    wire_lw  = max(lw * 0.50, 1.0) if is_tiny else lw * 0.50
    wink_lw  = max(lw * 0.75, 1.2) if is_tiny else lw * 0.75
    smile_lw = max(lw * 0.80, 1.2) if is_tiny else lw * 0.80

    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    if transparent:
        fig.patch.set_alpha(0.0)
    ax.set_aspect("equal"); ax.axis("off")
    fig.canvas.draw()

    # Face outline
    fo = draw_face_patch(
        width=FACE_SIZE, height=FACE_SIZE, angle_top_to_bottom=0,
        rounding_top=FACE_ROUND_TOP, rounding_bottom=FACE_ROUND_BOT,
        cx=0.5, cy=0.5, facecolor="none", edgecolor=LINE_COLOR,
        lw=lw, zorder=20)
    ax.add_patch(fo)

    # Brain/face fills — _px params unchanged since DPI is the same
    draw_face_brain(
        ax, fo, cx=0.5, cy=0.5, y_pos_px=30, curvature_px=-55,
        curve_dir="down", fill_pad_px=0, sep_n=700, boundary_interp=220,
        lw=lw, brain_color=BRAIN_COLOR, face_color=FACE_COLOR,
        line_color=LINE_COLOR)

    # Lens shadow — skip tiny
    if not is_tiny:
        c = tuple((np.array([0.5-1.22,0.5]) + np.array([0.5-1.20,0.5])) / 2)
        add_circle_lens_shadow(
            ax, face_outline=fo, circle_center=c, radius=1.52,
            brain_shadow_color=BRAIN_BAND[0], face_shadow_color=FACE_BAND[0],
            alpha=0.035, region="outside", prefer="right", zorder=28)

    # Circuits — skip tiny
    if not is_tiny:
        for sh in [-0.02, -0.30]:
            add_one_brain_circuit_from_brain_top(
                ax, fo, center_shift=sh, length_px=80, node_r=NODE_R,
                node_facecolor="#FFFFFF", node_edgecolor=NODE_EDGE,
                wire_lw=wire_lw, wire_color=WIRE_COLOR, zorder=40)
        for sh, bd in [(-0.02, -1), (0.20, +1)]:
            add_rect_bent_double_node_from_brain_top(
                ax, fo, center_shift=sh, length_px=75, bend_px=40,
                bend_at=0.7, bend_dir=bd, corner_px=12, corner_n=7,
                node1_on="vertical1", node1_u=0.58,
                node_facecolor1=NODE_LED, node_facecolor2=NODE_METAL,
                node_edgecolor=NODE_EDGE, node_r=NODE_R,
                wire_color=WIRE_COLOR, wire_lw=wire_lw, zorder=40)

    # Eyes
    draw_eyes(
        ax, fo, cx=0.5, cy=EYES_CY, open_left=False, open_right=True,
        eye_dx=0.19, eye_dy=-0.02, r=EYE_R, wink_lw=wink_lw,
        outline_color=LINE_COLOR, iris_color_left=EYE_IRIS,
        iris_color_right=EYE_IRIS, ring_ratio=0.10, iris_ratio=0.38,
        pupil_ratio=EYE_PUPIL_RATIO, spec_ratio=EYE_SPEC_RATIO,
        spec_offset=(0.30, 0.34), iris_offset_y=0.0,
        gaze_left=(1.0, 0.0), gaze_right=(-1.0, 0.0),
        gaze_strength=EYE_GAZE_STRENGTH, aa_shrink=0.995)

    # Smile
    add_smile(
        ax, fo, cx=0.5, cy=0.5, y=SMILE_Y, w=SMILE_W, h=SMILE_H,
        theta1=210, theta2=330, lw=smile_lw, color=LINE_COLOR)

    # Background circle
    background_circle(
        ax, center=(0.5, 0.5), radius=0.60, color=BACKGROUND_COLOR,
        alpha=0.90, edgecolor="#0B0F14", lw=bglw, zorder=0)

    # Top outline
    top = draw_face_patch(
        width=FACE_SIZE, height=FACE_SIZE, angle_top_to_bottom=0,
        rounding_top=FACE_ROUND_TOP, rounding_bottom=FACE_ROUND_BOT,
        cx=0.5, cy=0.5, facecolor="none", edgecolor=LINE_COLOR,
        lw=lw, zorder=1000)
    ax.add_patch(top)
    ax.relim(); ax.autoscale_view(); ax.margins(0.06)
    return fig


def fig_to_pil(fig, target_px):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0.02, transparent=False)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    if img.size != (target_px, target_px):
        img = img.resize((target_px, target_px), Image.LANCZOS)
    return img


# ═══════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════
TARGETS = {
    "favicon-16": 16, "favicon-32": 32, "favicon-48": 48,
    "apple-touch-icon-180": 180, "android-192": 192, "android-512": 512,
    "logo-64": 64, "logo-128": 128, "logo-256": 256,
    "logo-512": 512, "logo-1024": 1024,
}
ICO_SIZES = [16, 32, 48]

def main():
    out_dir = os.path.join(SCRIPT_DIR, "logo_output")
    os.makedirs(out_dir, exist_ok=True)
    images = {}
    for name, px in sorted(TARGETS.items(), key=lambda x: x[1]):
        print(f"  {name} ({px}×{px}) …")
        fig = render_logo(px)
        img = fig_to_pil(fig, px)
        img.save(os.path.join(out_dir, f"{name}.png"))
        images[px] = img

    ico = [images[s] for s in ICO_SIZES if s in images]
    if ico:
        ico[0].save(os.path.join(out_dir, "favicon.ico"), format="ICO",
                    sizes=[(s,s) for s in ICO_SIZES], append_images=ico[1:])
        print(f"  favicon.ico ({', '.join(map(str, ICO_SIZES))} px)")
    print(f"\nDone → {out_dir}/")

if __name__ == "__main__":
    main()
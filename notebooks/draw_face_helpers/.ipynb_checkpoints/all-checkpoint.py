import numpy as np
import matplotlib.lines as mlines
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path

from .geometry import _px_to_data_dx, _px_to_data_dy
from .separator import build_separator


def add_one_brain_circuit_from_brain_top(ax, face_patch,
                                        center_shift=0.0,   # fraction of brain width
                                        length_px=78,       # pixels downward from the top edge
                                        node_r=0.020,
                                        node_facecolor="white",
                                        node_edgecolor="black",
                                        wire_color="black",
                                        wire_lw=7,          # points
                                        zorder=26,
                                        cap_safety=1.10):
    """
    Same behavior: takes whatever you pass (now face_outline) and internally treats it
    as the brain region clip (previously 'brain_fill').
    """

    def top_y_at_x_from_poly(poly_xy, xq, eps=1e-9):
        if not np.allclose(poly_xy[0], poly_xy[-1]):
            poly_xy = np.vstack([poly_xy, poly_xy[0]])

        ys = []
        for (x1, y1), (x2, y2) in zip(poly_xy[:-1], poly_xy[1:]):
            dx = x2 - x1

            if abs(dx) <= eps:
                if abs(xq - x1) <= eps:
                    ys.append(float(y1))
                    ys.append(float(y2))
                continue

            if (min(x1, x2) - eps) <= xq <= (max(x1, x2) + eps):
                t = (xq - x1) / dx
                if (-eps) <= t <= (1.0 + eps):
                    ys.append(float(y1 + t * (y2 - y1)))

        if not ys:
            return None
        return max(ys)

    face_outline = face_patch

    brain_path = face_outline.get_path().transformed(face_outline.get_patch_transform())
    polys = brain_path.to_polygons(closed_only=True)
    if not polys:
        return None, None
    poly = max(polys, key=len)

    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    W = float(xmax - xmin)

    x_center = 0.5 * (xmin + xmax)
    x = float(x_center + float(center_shift) * W)

    y_top = top_y_at_x_from_poly(poly, x)
    if y_top is None:
        return None, None

    cap_radius_px = (float(wire_lw) * ax.figure.dpi / 72.0) * 0.5
    cap_dy = abs(_px_to_data_dy(ax, cap_radius_px * float(cap_safety)))
    y_start = float(y_top + cap_dy)

    dy = abs(_px_to_data_dy(ax, float(length_px)))
    y_end = float(y_top - dy)

    wire = mlines.Line2D([x, x], [y_start, y_end],
                         lw=wire_lw, color=wire_color,
                         solid_capstyle="round",
                         zorder=zorder)
    wire.set_clip_path(face_outline)
    ax.add_line(wire)

    node = Circle((x, y_end), node_r,
                  facecolor=node_facecolor,
                  edgecolor=node_edgecolor,
                  linewidth=float(wire_lw) * 0.55,
                  zorder=zorder + 1)
    node.set_clip_path(face_outline)
    ax.add_patch(node)

    return node, wire


def add_bent_brain_circuit_from_face_outline(ax, face_patch,
                                            cx=0.5, cy=0.5,
                                            y_pos_px=15,
                                            curvature_px=35,
                                            boundary_interp=220,
                                            sep_n=700,
                                            fill_pad_px=2,
                                            center_shift=0.0,
                                            length_px=90,
                                            bend_px=28,
                                            bend_at=0.45,
                                            node_r=0.020,
                                            node_facecolor="white",
                                            node_edgecolor="black",
                                            wire_color="black",
                                            wire_lw=7,
                                            zorder=26,
                                            cap_safety=1.10):
    """
    Bent brain circuit that RECEIVES face_patch (face_outline) and clips wire+node to the computed brain region.
    """

    y_sep, sep_pts, L, R, poly = build_separator(
        ax, face_patch, cx=cx, cy=cy,
        y_pos_px=y_pos_px, curvature_px=curvature_px,
        n=sep_n, boundary_interp=boundary_interp
    )

    pad_dy = _px_to_data_dy(ax, fill_pad_px) if fill_pad_px else 0.0
    if pad_dy != 0.0:
        sep_top = sep_pts.copy()
        sep_top[:, 1] -= abs(pad_dy)
    else:
        sep_top = sep_pts

    def arc_points(A, B):
        iA, iB = A["i"], B["i"]
        pts = [np.array([A["x"], A["y"]], dtype=float)]
        j = iA + 1
        N = len(poly) - 1
        while True:
            pts.append(poly[j % N].astype(float))
            if (j % N) == (iB % N):
                break
            j += 1
        pts[-1] = np.array([B["x"], B["y"]], dtype=float)
        return np.vstack(pts)

    arc_LR = arc_points(L, R)
    arc_RL = arc_points(R, L)

    top_arc = arc_LR if (np.mean(arc_LR[:, 1]) >= np.mean(arc_RL[:, 1])) else arc_RL
    top_poly = np.vstack([top_arc, sep_top[::-1, :]])

    verts = [tuple(top_poly[0])]
    codes = [Path.MOVETO]
    for p in top_poly[1:]:
        verts.append(tuple(p))
        codes.append(Path.LINETO)
    verts.append((0.0, 0.0))
    codes.append(Path.CLOSEPOLY)

    brain_clip = PathPatch(Path(verts, codes), facecolor="none", edgecolor="none")
    brain_clip.set_clip_path(face_patch)
    brain_clip.set_alpha(0.0)
    brain_clip.set_zorder(-999)
    ax.add_patch(brain_clip)

    xmin, ymin = top_poly.min(axis=0)
    xmax, ymax = top_poly.max(axis=0)
    W = float(xmax - xmin)
    x_center = 0.5 * (xmin + xmax)
    x0 = float(x_center + float(center_shift) * W)

    def top_y_at_x(polyline_xy, xq, eps=1e-9):
        ys = []
        for (x1, y1), (x2, y2) in zip(polyline_xy[:-1], polyline_xy[1:]):
            dx = x2 - x1
            if abs(dx) <= eps:
                if abs(xq - x1) <= eps:
                    ys.append(float(y1)); ys.append(float(y2))
                continue
            if (min(x1, x2) - eps) <= xq <= (max(x1, x2) + eps):
                t = (xq - x1) / dx
                if (-eps) <= t <= (1.0 + eps):
                    ys.append(float(y1 + t * (y2 - y1)))
        return max(ys) if ys else None

    y_top = top_y_at_x(top_arc, x0)
    if y_top is None:
        return None, None

    cap_radius_px = (float(wire_lw) * ax.figure.dpi / 72.0) * 0.5
    cap_dy = abs(_px_to_data_dy(ax, cap_radius_px * float(cap_safety)))
    y0 = float(y_top + cap_dy)

    dy_total = abs(_px_to_data_dy(ax, float(length_px)))
    dx_bend = _px_to_data_dx(ax, float(bend_px))

    y1 = float(y_top - dy_total * float(bend_at))
    x1 = float(x0 + dx_bend)

    y2 = float(y_top - dy_total)
    x2 = float(x1)

    wire = mlines.Line2D([x0, x1, x2], [y0, y1, y2],
                         lw=wire_lw, color=wire_color,
                         solid_capstyle="round",
                         solid_joinstyle="round",
                         zorder=zorder)
    wire.set_clip_path(brain_clip)
    ax.add_line(wire)

    node = Circle((x2, y2), node_r,
                  facecolor=node_facecolor,
                  edgecolor=node_edgecolor,
                  linewidth=float(wire_lw) * 0.55,
                  zorder=zorder + 1)
    node.set_clip_path(brain_clip)
    ax.add_patch(node)

    return node, wire


def add_s_double_node_circuit_from_brain_top(ax, face_patch,
                                             center_shift=0.0,
                                             length_px=95,
                                             amp_px=18,
                                             waves=1.0,
                                             n=220,
                                             node_t1=0.45,
                                             node_t2=0.85,
                                             node_r=0.020,
                                             node_facecolor="white",
                                             node_edgecolor="black",
                                             wire_color="black",
                                             wire_lw=7,
                                             zorder=26,
                                             cap_safety=1.10):
    """
    Same behavior: takes whatever you pass (now face_outline) and internally treats it
    as the brain region clip (previously 'brain_fill').
    """

    def top_y_at_x_from_poly(poly_xy, xq, eps=1e-9):
        if not np.allclose(poly_xy[0], poly_xy[-1]):
            poly_xy = np.vstack([poly_xy, poly_xy[0]])

        ys = []
        for (x1, y1), (x2, y2) in zip(poly_xy[:-1], poly_xy[1:]):
            dx = x2 - x1

            if abs(dx) <= eps:
                if abs(xq - x1) <= eps:
                    ys.append(float(y1))
                    ys.append(float(y2))
                continue

            if (min(x1, x2) - eps) <= xq <= (max(x1, x2) + eps):
                t = (xq - x1) / dx
                if (-eps) <= t <= (1.0 + eps):
                    ys.append(float(y1 + t * (y2 - y1)))

        return max(ys) if ys else None

    face_outline = face_patch

    brain_path = face_outline.get_path().transformed(face_outline.get_patch_transform())
    polys = brain_path.to_polygons(closed_only=True)
    if not polys:
        return None, None, None

    poly = max(polys, key=len)
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    W = float(xmax - xmin)

    x_center = 0.5 * (xmin + xmax)
    x0 = float(x_center + float(center_shift) * W)

    y_top = top_y_at_x_from_poly(poly, x0)
    if y_top is None:
        return None, None, None

    cap_radius_px = (float(wire_lw) * ax.figure.dpi / 72.0) * 0.5
    cap_dy = abs(_px_to_data_dy(ax, cap_radius_px * float(cap_safety)))
    y_start = float(y_top + cap_dy)

    dy_total = abs(_px_to_data_dy(ax, float(length_px)))
    dx_amp = _px_to_data_dx(ax, float(amp_px))

    t = np.linspace(0.0, 1.0, int(n))
    xs = x0 + dx_amp * np.sin(2.0 * np.pi * float(waves) * t)
    ys = y_start - dy_total * t

    wire = mlines.Line2D(xs, ys,
                         lw=wire_lw, color=wire_color,
                         solid_capstyle="round",
                         solid_joinstyle="round",
                         zorder=zorder)
    wire.set_clip_path(face_outline)
    ax.add_line(wire)

    def add_node_at(tt):
        tt = float(np.clip(tt, 0.0, 1.0))
        i = int(round(tt * (len(t) - 1)))
        node = Circle((float(xs[i]), float(ys[i])), node_r,
                      facecolor=node_facecolor,
                      edgecolor=node_edgecolor,
                      linewidth=float(wire_lw) * 0.55,
                      zorder=zorder + 1)
        node.set_clip_path(face_outline)
        ax.add_patch(node)
        return node

    node1 = add_node_at(node_t1)
    node2 = add_node_at(node_t2)

    return node1, node2, wire


def add_rect_bent_double_node_from_brain_top(ax, face_patch,
                                            center_shift=0.0,
                                            length_px=80,
                                            bend_px=22,
                                            bend_at=0.35,
                                            bend_dir=+1,
                                            node1_on="vertical1",   # NEW: "vertical1" | "horizontal" | "vertical2"
                                            node1_u=0.6,            # NEW: 0..1 along that segment
                                            node_facecolor1="#4aa3c7",
                                            node_facecolor2="#4aa3c7",
                                            node_edgecolor="black",
                                            node_r=0.020,
                                            wire_color="black",
                                            wire_lw=7,
                                            zorder=26,
                                            cap_safety=1.10,
                                            corner_px=6,
                                            corner_n=20):
    """
    Rect bent wire with two nodes:
    - node2 is always at the end of the wire
    - bend_dir mirrors the bend (+1 right, -1 left)
    - node1 placement is controllable by segment (node1_on) + fraction (node1_u)
    """

    def top_y_at_x_from_poly(poly_xy, xq, eps=1e-9):
        if not np.allclose(poly_xy[0], poly_xy[-1]):
            poly_xy = np.vstack([poly_xy, poly_xy[0]])
        ys = []
        for (x1, y1), (x2, y2) in zip(poly_xy[:-1], poly_xy[1:]):
            dx = x2 - x1
            if abs(dx) <= eps:
                if abs(xq - x1) <= eps:
                    ys.append(float(y1)); ys.append(float(y2))
                continue
            if (min(x1, x2) - eps) <= xq <= (max(x1, x2) + eps):
                t = (xq - x1) / dx
                if (-eps) <= t <= (1.0 + eps):
                    ys.append(float(y1 + t * (y2 - y1)))
        return max(ys) if ys else None

    face_outline = face_patch

    brain_path = face_outline.get_path().transformed(face_outline.get_patch_transform())
    polys = brain_path.to_polygons(closed_only=True)
    if not polys:
        return None, None, None

    poly = max(polys, key=len)
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    W = float(xmax - xmin)

    x_center = 0.5 * (xmin + xmax)
    x0 = float(x_center + float(center_shift) * W)

    y_top = top_y_at_x_from_poly(poly, x0)
    if y_top is None:
        return None, None, None

    cap_radius_px = (float(wire_lw) * ax.figure.dpi / 72.0) * 0.5
    cap_dy = abs(_px_to_data_dy(ax, cap_radius_px * float(cap_safety)))
    y0 = float(y_top + cap_dy)

    dy_total = abs(_px_to_data_dy(ax, float(length_px)))

    dx_step = _px_to_data_dx(ax, float(bend_px))
    s = 1.0 if float(bend_dir) >= 0 else -1.0   # +right, -left
    dx_step *= s

    y_corner = float(y_top - dy_total * float(bend_at))
    x0_base = float(x0)
    x_right = float(x0 + dx_step)
    y_end   = float(y_top - dy_total)

    r = max(0.0, float(_px_to_data_dx(ax, float(corner_px))))

    max_r1 = abs(y_corner - y0) * 0.45
    max_r2 = abs(x_right - x0_base) * 0.45
    max_r3 = abs(y_end - y_corner) * 0.45
    r = min(r, max_r1, max_r2, max_r3)

    # --- node1 position (stable under mirroring)
    node1_on_ = str(node1_on).lower().strip()
    u = float(np.clip(node1_u, 0.0, 1.0))

    if node1_on_ == "vertical1":
        p1 = (float(x0_base), float(y0 + (y_corner - y0) * u))
    elif node1_on_ == "horizontal":
        p1 = (float(x0_base + (x_right - x0_base) * u), float(y_corner))
    elif node1_on_ == "vertical2":
        p1 = (float(x_right), float(y_corner + (y_end - y_corner) * u))
    else:
        raise ValueError("node1_on must be 'vertical1', 'horizontal', or 'vertical2'")

    # --- build rounded polyline
    xA, yA = float(x0_base), float(y0)
    xB, yB = float(x0_base), float(y_corner + r)

    c1x, c1y = float(x0_base + s * r), float(y_corner + r)
    xD, yD   = float(x_right - s * r), float(y_corner)
    c2x, c2y = float(x_right - s * r), float(y_corner - r)

    xF, yF = float(x_right), float(y_end)

    if r > 0:
        if s > 0:
            ang1 = np.linspace(np.pi, 1.5*np.pi, int(corner_n))
            ang2 = np.linspace(0.5*np.pi, 0.0, int(corner_n))
        else:
            ang1 = np.linspace(0.0, -0.5*np.pi, int(corner_n))
            ang2 = np.linspace(0.5*np.pi, np.pi, int(corner_n))

        arc1x = c1x + r * np.cos(ang1)
        arc1y = c1y + r * np.sin(ang1)

        arc2x = c2x + r * np.cos(ang2)
        arc2y = c2y + r * np.sin(ang2)
    else:
        arc1x = np.array([x0_base], dtype=float); arc1y = np.array([y_corner], dtype=float)
        arc2x = np.array([x_right], dtype=float); arc2y = np.array([y_corner], dtype=float)

    xs = np.concatenate([[xA, xB], arc1x, [xD], arc2x, [xF]]).astype(float)
    ys = np.concatenate([[yA, yB], arc1y, [yD], arc2y, [yF]]).astype(float)

    wire = mlines.Line2D(xs, ys,
                         lw=wire_lw, color=wire_color,
                         solid_capstyle="round",
                         solid_joinstyle="round",
                         zorder=zorder)
    wire.set_clip_path(face_outline)
    ax.add_line(wire)

    # node2 always at the end of the wire
    p2 = (float(xs[-1]), float(ys[-1]))

    node1 = Circle(p1, node_r,
                   facecolor=node_facecolor1,
                   edgecolor=node_edgecolor,
                   linewidth=float(wire_lw) * 0.55,
                   zorder=zorder + 1)
    node1.set_clip_path(face_outline)
    ax.add_patch(node1)

    node2 = Circle(p2, node_r,
                   facecolor=node_facecolor2,
                   edgecolor=node_edgecolor,
                   linewidth=float(wire_lw) * 0.55,
                   zorder=zorder + 1)
    node2.set_clip_path(face_outline)
    ax.add_patch(node2)

    return node1, node2, wire

from matplotlib.patches import Circle, Arc
import numpy as np


def add_eye(ax, cx, cy,
            open_eye=True,
            r=0.06,
            wink_lw=10,
            outline_color="black",
            iris_color="#49a37d",
            ring_ratio=0.10,
            iris_ratio=0.38,
            pupil_ratio=0.45,
            spec_ratio=0.18,
            spec_offset=(0.25, 0.25),
            iris_offset_y=0.0,
            gaze=(0.0, 0.0),          # NEW: (gx, gy) in [-1, 1] on an imaginary circle
            gaze_strength=0.35,       # NEW: how far pupil/iris can move (fraction of inner radius)
            aa_shrink=0.995,
            zorder=40):

    artists = []
    if open_eye:
        dy = iris_offset_y * r

        sclera = Circle((cx, cy), radius=r, facecolor="white", edgecolor="none", zorder=zorder)
        ax.add_patch(sclera); artists.append(sclera)

        rim = Circle((cx, cy), radius=r, facecolor="none", edgecolor=outline_color,
                     linewidth=wink_lw, zorder=zorder+1)
        ax.add_patch(rim); artists.append(rim)

        inner_r = r * (1.0 - ring_ratio)
        clip_r = inner_r * aa_shrink
        inner_clip = Circle((cx, cy), radius=clip_r, transform=ax.transData)

        # --- NEW: gaze offset in data coords
        gx, gy = gaze
        gx = float(np.clip(gx, -1.0, 1.0))
        gy = float(np.clip(gy, -1.0, 1.0))

        # keep within the unit circle
        norm = (gx*gx + gy*gy) ** 0.5
        if norm > 1.0 and norm > 0:
            gx /= norm
            gy /= norm

        # max offset proportional to inner radius
        max_off = float(gaze_strength) * float(inner_r)
        offx = gx * max_off
        offy = gy * max_off

        ri = inner_r * float(iris_ratio)
        iris = Circle((cx + offx, cy + dy + offy), radius=ri,
                      facecolor=iris_color, edgecolor="none", zorder=zorder+2)
        iris.set_clip_path(inner_clip)
        ax.add_patch(iris); artists.append(iris)

        rp = ri * float(pupil_ratio)
        pupil = Circle((cx + offx, cy + dy + offy), radius=rp,
                       facecolor="black", edgecolor="none", zorder=zorder+3)
        pupil.set_clip_path(inner_clip)
        ax.add_patch(pupil); artists.append(pupil)

        rs = ri * float(spec_ratio)
        ox, oy = spec_offset
        spec = Circle((cx - ox*ri + offx, cy + dy + oy*ri + offy), radius=rs,
                      facecolor="white", edgecolor="none", zorder=zorder+4)
        spec.set_clip_path(inner_clip)
        ax.add_patch(spec); artists.append(spec)

        return artists

    wink = Arc((cx, cy), width=2*r, height=1.15*r,
               angle=0, theta1=200, theta2=340,
               color=outline_color, linewidth=wink_lw,
               capstyle="round", zorder=zorder)
    ax.add_patch(wink); artists.append(wink)
    return artists


def draw_eyes(ax, face_patch,
              cx=0.5, cy=0.5,
              open_left=True, open_right=True,
              eye_dx=0.17,
              eye_dy=-0.05,
              r=0.06,
              wink_lw=10,
              outline_color="black",
              iris_color_left="#49a37d",
              iris_color_right="#49a37d",
              ring_ratio=0.10,
              iris_ratio=0.38,
              pupil_ratio=0.45,
              spec_ratio=0.18,
              spec_offset=(0.30, 0.34),
              iris_offset_y=0.0,
              gaze_left=(0.0, 0.0),       # NEW
              gaze_right=(0.0, 0.0),      # NEW
              gaze_strength=0.35,         # NEW (shared)
              aa_shrink=0.995,
              zorder=40,
              clip=True):

    left_x, left_y = (cx - eye_dx, cy + eye_dy)
    right_x, right_y = (cx + eye_dx, cy + eye_dy)

    left_artists = add_eye(
        ax, left_x, left_y, open_eye=open_left,
        r=r, wink_lw=wink_lw,
        outline_color=outline_color,
        iris_color=iris_color_left,
        ring_ratio=ring_ratio,
        iris_ratio=iris_ratio,
        pupil_ratio=pupil_ratio,
        spec_ratio=spec_ratio,
        spec_offset=spec_offset,
        iris_offset_y=iris_offset_y,
        gaze=gaze_left,
        gaze_strength=gaze_strength,
        aa_shrink=aa_shrink,
        zorder=zorder
    )

    right_artists = add_eye(
        ax, right_x, right_y, open_eye=open_right,
        r=r, wink_lw=wink_lw,
        outline_color=outline_color,
        iris_color=iris_color_right,
        ring_ratio=ring_ratio,
        iris_ratio=iris_ratio,
        pupil_ratio=pupil_ratio,
        spec_ratio=spec_ratio,
        spec_offset=spec_offset,
        iris_offset_y=iris_offset_y,
        gaze=gaze_right,
        gaze_strength=gaze_strength,
        aa_shrink=aa_shrink,
        zorder=zorder
    )

    if clip:
        for a in left_artists + right_artists:
            a.set_clip_path(face_patch)

    return left_artists, right_artists
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def _ellipse_path(cx, cy, width, height, n=240):
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    x = cx + 0.5 * float(width) * np.cos(t)
    y = cy + 0.5 * float(height) * np.sin(t)

    verts = [(float(x[0]), float(y[0]))]
    codes = [Path.MOVETO]
    for xi, yi in zip(x[1:], y[1:]):
        verts.append((float(xi), float(yi)))
        codes.append(Path.LINETO)
    verts.append((0.0, 0.0))
    codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)


# ----------------------------
# Face patch: "rect" (your current rounded skewed rectangle) OR "oval" (ellipse)
# ----------------------------
def draw_face_patch(width, height, angle_top_to_bottom=0,
                    rounding_top=0.0, rounding_bottom=0.0,
                    cx=0.5, cy=0.5,
                    facecolor="none", edgecolor="black", lw=10,
                    zorder=20,
                    shape="rect",          # NEW: "rect" | "oval"
                    oval_n=240):            # NEW: ellipse smoothness

    shape = str(shape).lower().strip()
    if shape not in ("rect", "oval", "ellipse"):
        raise ValueError("shape must be 'rect' or 'oval' (aka 'ellipse')")

    # ---- OVAL MODE
    if shape in ("oval", "ellipse"):
        path = _ellipse_path(cx, cy, width, height, n=oval_n)
        return PathPatch(
            path,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            joinstyle="round",
            capstyle="round",
            zorder=zorder,
        )

    # ---- RECT MODE (your current implementation)
    theta = np.deg2rad(angle_top_to_bottom)
    dx = height * np.tan(theta)

    w_top = width
    w_bot = width + 2 * dx

    y_top = cy + height / 2
    y_bot = cy - height / 2

    tl = np.array([cx - w_top / 2, y_top])
    tr = np.array([cx + w_top / 2, y_top])
    br = np.array([cx + w_bot / 2, y_bot])
    bl = np.array([cx - w_bot / 2, y_bot])

    corners = [tl, tr, br, bl]  # clockwise
    r_corner = np.array([rounding_top, rounding_top, rounding_bottom, rounding_bottom], dtype=float)

    edge_len = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    for i in range(4):
        adj_min = min(edge_len[i - 1], edge_len[i])
        r_corner[i] = max(0.0, min(float(r_corner[i]), 0.49 * float(adj_min)))

    p_in, p_out = [], []
    for i in range(4):
        p = corners[i]
        p_prev = corners[i - 1]
        p_next = corners[(i + 1) % 4]
        r = float(r_corner[i])

        u_prev = (p_prev - p); u_prev /= np.linalg.norm(u_prev)
        u_next = (p_next - p); u_next /= np.linalg.norm(u_next)

        p_in.append(p + u_prev * r)
        p_out.append(p + u_next * r)

    verts, codes = [], []
    verts.append(tuple(p_out[0])); codes.append(Path.MOVETO)

    for i in [1, 2, 3]:
        verts.append(tuple(p_in[i]));     codes.append(Path.LINETO)
        verts.append(tuple(corners[i]));  codes.append(Path.CURVE3)
        verts.append(tuple(p_out[i]));    codes.append(Path.CURVE3)

    verts.append(tuple(p_in[0]));     codes.append(Path.LINETO)
    verts.append(tuple(corners[0]));  codes.append(Path.CURVE3)
    verts.append(tuple(p_out[0]));    codes.append(Path.CURVE3)

    verts.append((0.0, 0.0)); codes.append(Path.CLOSEPOLY)

    return PathPatch(
        Path(verts, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        joinstyle="round",
        capstyle="round",
        zorder=zorder
    )


# ----------------------------
# Get dense face boundary polygon in data coords
# ----------------------------
def _face_boundary_polygon(face_patch, interp=120):
    path_data = face_patch.get_path().transformed(face_patch.get_patch_transform())
    path_dense = path_data.interpolated(interp)
    polys = path_dense.to_polygons(closed_only=True)
    if not polys:
        raise ValueError("Could not extract polygon from face patch.")
    return max(polys, key=lambda a: a.shape[0])

import numpy as np
import matplotlib.lines as mlines
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from .separator import build_separator
from .geometry import _px_to_data_dy

# ----------------------------
# Draw fills + separator line
# ----------------------------
def draw_face_brain(ax, face_patch,
                    cx=0.5, cy=0.5,
                    y_pos_px=15,
                    curvature_px=35,
                    curve_dir="down",
                    sep_n=600,
                    boundary_interp=200,
                    fill_pad_px=0,
                    lw=10,
                    brain_color="#cfe8ff",
                    face_color="#ffe0cc",
                    line_color="black",
                    z_fill=10,
                    z_line=30):

    y_sep, sep_pts, L, R, poly = build_separator(
        ax, face_patch, cx=cx, cy=cy,
        y_pos_px=y_pos_px, curvature_px=curvature_px,
        n=sep_n, boundary_interp=boundary_interp,curve_dir=curve_dir,
    )

    pad_dy = _px_to_data_dy(ax, fill_pad_px) if fill_pad_px else 0.0
    if pad_dy != 0.0:
        sep_top = sep_pts.copy(); sep_top[:, 1] -= abs(pad_dy)
        sep_bot = sep_pts.copy(); sep_bot[:, 1] += abs(pad_dy)
    else:
        sep_top = sep_pts
        sep_bot = sep_pts

    def arc_points(A, B):
        iA, iB = A["i"], B["i"]
        pts = [np.array([A["x"], A["y"]], dtype=float)]
        j = iA + 1
        N = len(poly) - 1
        while True:
            pts.append(poly[j % N].astype(float))
            if (j % N) == (iB % N):
                break
            j += 1
        pts[-1] = np.array([B["x"], B["y"]], dtype=float)
        return np.vstack(pts)

    arc_LR = arc_points(L, R)
    arc_RL = arc_points(R, L)

    if np.mean(arc_LR[:, 1]) >= np.mean(arc_RL[:, 1]):
        top_arc, bot_arc = arc_LR, arc_RL
    else:
        top_arc, bot_arc = arc_RL, arc_LR

    top_poly = np.vstack([top_arc, sep_top[::-1, :]])
    bot_poly = np.vstack([sep_bot, bot_arc[::-1, :]])

    def poly_to_patch(pts, color, z):
        verts = [tuple(pts[0])]
        codes = [Path.MOVETO]
        for p in pts[1:]:
            verts.append(tuple(p))
            codes.append(Path.LINETO)
        verts.append((0.0, 0.0)); codes.append(Path.CLOSEPOLY)
        patch = PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=z)
        patch.set_clip_path(face_patch)
        ax.add_patch(patch)
        return patch

    face_fill = poly_to_patch(bot_poly, face_color, z_fill)

    # NOTE: keep return order the same, but internally name this face_outline (per your request)
    face_outline = poly_to_patch(top_poly, brain_color, z_fill + 1)

    line = mlines.Line2D(sep_pts[:, 0], sep_pts[:, 1],
                         lw=lw, color=line_color,
                         solid_capstyle="round",
                         zorder=z_line)
    line.set_clip_path(face_patch)
    ax.add_line(line)

    # return stays the same: (brain_fill, face_fill, line)
    return face_outline, face_fill, line

# ----------------------------
# px->data conversions
# ----------------------------

import numpy as np
from matplotlib.patches import Circle



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
    You fully control the radius.

    Returns the Circle patch.
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





def _px_to_data_dy(ax, px):
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0.0, 0.0))
    x1, y1 = inv.transform((0.0, float(px)))
    return (y1 - y0)

def _px_to_data_dx(ax, px):
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0.0, 0.0))
    x1, y1 = inv.transform((float(px), 0.0))
    return (x1 - x0)

from matplotlib.patches import Arc

# ----------------------------
# Smile
# ----------------------------
def add_smile(ax, face_patch,
              cx=0.5, cy=0.5,
              y=-0.18,
              w=0.42,
              h=0.26,
              theta1=200, theta2=340,
              color="black",
              lw=10,
              zorder=45,
              clip=True):
    smile = Arc((cx, cy + y), width=w, height=h,
                angle=0, theta1=theta1, theta2=theta2,
                color=color, linewidth=lw, capstyle="round",
                zorder=zorder)
    if clip:
        smile.set_clip_path(face_patch)
    ax.add_patch(smile)
    return smile
import numpy as np
from .geometry import _px_to_data_dy
from .face import _face_boundary_polygon


def build_separator(ax, face_patch, cx=0.5, cy=0.5,
                    y_pos_px=15, curvature_px=35,
                    n=600, boundary_interp=200,
                    curve_dir="up"):   # NEW: "down" (smile) or "up" (sad)
    ax.figure.canvas.draw()

    dy = _px_to_data_dy(ax, y_pos_px)
    dcurve = abs(_px_to_data_dy(ax, curvature_px))
    y_sep = cy + dy

    curve_dir = str(curve_dir).lower().strip()
    if curve_dir in ("down", "smile", "convex", "u"):
        sign = -1.0
    elif curve_dir in ("up", "sad", "concave", "n"):
        sign = +1.0
    else:
        raise ValueError("curve_dir must be 'down' or 'up' (aliases: smile/sad)")

    poly = _face_boundary_polygon(face_patch, interp=boundary_interp)

    inter = []
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(poly[:-1], poly[1:])):
        if (y1 <= y_sep <= y2) or (y2 <= y_sep <= y1):
            dyseg = (y2 - y1)
            if abs(dyseg) < 1e-12:
                continue
            tseg = (y_sep - y1) / dyseg
            if 0.0 <= tseg <= 1.0:
                x = x1 + tseg * (x2 - x1)
                inter.append({"x": float(x), "y": float(y_sep), "i": i, "t": float(tseg)})

    if len(inter) < 2:
        y_min = float(np.min(poly[:, 1])); y_max = float(np.max(poly[:, 1]))
        raise ValueError(
            f"Separator y={y_sep:.4f} did not intersect face. "
            f"Face y-range=[{y_min:.4f}, {y_max:.4f}]."
        )

    inter = sorted(inter, key=lambda d: d["x"])
    inter2 = [inter[0]]
    for d in inter[1:]:
        if abs(d["x"] - inter2[-1]["x"]) > 1e-6:
            inter2.append(d)
    if len(inter2) < 2:
        raise ValueError("Intersections collapsed after dedupe.")

    L, R = inter2[0], inter2[-1]

    p0 = np.array([L["x"], y_sep], dtype=float)
    p2 = np.array([R["x"], y_sep], dtype=float)
    pm = 0.5 * (p0 + p2)

    # ctrl y determines curvature orientation
    ctrl = np.array([pm[0], pm[1] + sign * dcurve], dtype=float)

    t = np.linspace(0.0, 1.0, int(n))
    sep_pts = ((1 - t)[:, None] ** 2) * p0 + (2 * (1 - t)[:, None] * t[:, None]) * ctrl + (t[:, None] ** 2) * p2
    return y_sep, sep_pts, L, R, poly

import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# ----------------------------
# Face-follow crescent shading using x_center +/- n gating
# ----------------------------
def add_face_follow_crescent(ax, face_patch,
                             side="right",
                             color="#5b2cff",
                             alpha=0.22,
                             width_frac=0.10,
                             margin_frac=0.06,
                             smooth_window=11,
                             zorder=28):
    path = face_patch.get_path().transformed(face_patch.get_patch_transform())
    dense = path.interpolated(300).to_polygons(closed_only=True)
    if not dense:
        return None
    poly = max(dense, key=len)

    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    W = float(xmax - xmin)

    x_center = 0.5 * (xmin + xmax)
    n = float(margin_frac) * W

    side = side.lower().strip()
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    if side == "right":
        mask = poly[:, 0] > (x_center + n)
        inward_sign = -1.0







def add_circle_lens_shadow(ax,
                           face_outline,
                           circle_center=(0.5, 0.5),
                           radius=1.25,
                           brain_shadow_color="#5b2cff",
                           face_shadow_color="#5b2cff",
                           alpha=0.18,
                           poly_interp=600,
                           circle_n=1200,
                           zorder=28,
                           region="outside",   # "inside" or "outside"
                           prefer="right"):    # "right" or "left"
    """
    region="inside"  -> shades face ∩ circle  (your previous behavior)
    region="outside" -> shades face \ circle  (the one you want, to the right in your example)
    prefer selects which side if there are multiple candidate regions.
    """

    # ---- dense face polygon (data coords)
    path_data = face_outline.get_path().transformed(face_outline.get_patch_transform())
    poly_list = path_data.interpolated(int(poly_interp)).to_polygons(closed_only=True)
    if not poly_list:
        return None, None
    face_poly = max(poly_list, key=len)
    face_path = Path(face_poly)

    cx, cy = map(float, circle_center)
    r = float(radius)

    # ---- sample points on circle
    ang = np.linspace(0.0, 2.0 * np.pi, int(circle_n), endpoint=False)
    circle_pts = np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)]

    # points on circle that lie inside the face (this is the circle arc used as boundary)
    inside_face = face_path.contains_points(circle_pts)

    # ---- helper: all contiguous True runs on a circular boolean array
    def all_true_runs(mask):
        mask = np.asarray(mask, dtype=bool)
        n = len(mask)
        if n == 0 or not mask.any():
            return []

        # find transitions
        m = mask.astype(int)
        # indices where value changes
        changes = np.where(np.diff(np.r_[m, m[0]]) != 0)[0] + 1

        if len(changes) == 0:
            return [np.arange(n)]

        # build segments between changes
        segs = []
        start = changes[0]
        for end in changes[1:]:
            seg = np.arange(start, end) % n
            if mask[seg[0]]:
                segs.append(seg)
            start = end
        seg = np.arange(start, changes[0] + n) % n
        if mask[seg[0]]:
            segs.append(seg)

        # keep only “real” segments
        return [s for s in segs if len(s) >= 3]

    # circle arc candidates (usually 1, but keep it general)
    arc_runs = all_true_runs(inside_face)
    if not arc_runs:
        return None, None

    # ---- face boundary classification: inside vs outside circle
    d2 = (face_poly[:, 0] - cx) ** 2 + (face_poly[:, 1] - cy) ** 2
    inside_circle = d2 <= (r * r)

    if region.lower().strip() == "inside":
        face_mask = inside_circle
    elif region.lower().strip() == "outside":
        face_mask = ~inside_circle
    else:
        raise ValueError("region must be 'inside' or 'outside'")

    face_runs = all_true_runs(face_mask)
    if not face_runs:
        return None, None

    # ---- choose the best (face_run, arc_run) pair by side preference
    face_center_x = float(0.5 * (face_poly[:, 0].min() + face_poly[:, 0].max()))
    want_right = (prefer.lower().strip() == "right")

    best = None
    best_score = None

    for fr in face_runs:
        face_edge_pts = face_poly[fr]
        A = face_edge_pts[0]
        B = face_edge_pts[-1]

        for ar in arc_runs:
            arc_pts = circle_pts[ar]

            # pick arc direction that matches endpoints better
            s1 = np.linalg.norm(arc_pts[0] - A) + np.linalg.norm(arc_pts[-1] - B)
            s2 = np.linalg.norm(arc_pts[-1] - A) + np.linalg.norm(arc_pts[0] - B)
            arc_use = arc_pts if (s1 <= s2) else arc_pts[::-1]

            # stitch polygon: face segment + reversed arc (to close)
            pts = np.vstack([face_edge_pts, arc_use[::-1]])

            # score by centroid-x (prefer rightmost or leftmost)
            cx_poly = float(np.mean(pts[:, 0]))
            score = (cx_poly - face_center_x)
            score = score if want_right else (-score)

            if (best is None) or (score > best_score):
                best = pts
                best_score = score

    if best is None or len(best) < 10:
        return None, None

    lens_pts = best

    # ---- build patch from stitched polygon
    verts = [tuple(lens_pts[0])]
    codes = [Path.MOVETO]
    for p in lens_pts[1:]:
        verts.append(tuple(p))
        codes.append(Path.LINETO)
    verts.append((0.0, 0.0))
    codes.append(Path.CLOSEPOLY)
    lens_path = Path(verts, codes)

    # draw twice, clip separately so you can color brain vs face differently
    brain_shadow = PathPatch(lens_path, facecolor=brain_shadow_color, edgecolor="none",
                             alpha=alpha, zorder=zorder)
    brain_shadow.set_clip_path(face_outline)
    ax.add_patch(brain_shadow)

    face_shadow = PathPatch(lens_path, facecolor=face_shadow_color, edgecolor="none",
                            alpha=alpha, zorder=zorder)
    face_shadow.set_clip_path(face_outline)
    ax.add_patch(face_shadow)

    return brain_shadow, face_shadow

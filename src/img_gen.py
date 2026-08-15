"""Generates a seamlessly tileable gothic pattern into assets/bg.png."""

import os

import cv2
import numpy as np

SIZE = 512
HALF = SIZE // 2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bg.png")

# BGR
BG = (5, 2, 7)
WASH = (44, 8, 24)
HATCH = (28, 6, 16)
DEEP = (104, 14, 48)
VIOLET = (196, 34, 118)
AZURE = (255, 110, 58)
GLINT = (255, 198, 172)


def wrap_draw(canvas, pts, color, thickness=1, closed=True, fill=False):
    """Draw a polyline 9 times (3x3 offsets) so anything crossing an edge
    reappears on the opposite edge. This is what makes the tile seamless."""
    pts = np.asarray(pts, np.float32)
    for dx in (-SIZE, 0, SIZE):
        for dy in (-SIZE, 0, SIZE):
            q = np.round(pts + (dx, dy)).astype(np.int32)
            if q[:, 0].max() < 0 or q[:, 1].max() < 0:
                continue
            if q[:, 0].min() >= SIZE or q[:, 1].min() >= SIZE:
                continue
            if fill:
                cv2.fillPoly(canvas, [q], color, cv2.LINE_AA)
            else:
                cv2.polylines(canvas, [q], closed, color, thickness, cv2.LINE_AA)


def stroke(canvas, pts, closed=False, layers=((DEEP, 3), (VIOLET, 1))):
    for color, t in layers:
        wrap_draw(canvas, pts, color, thickness=t, closed=closed)


def solid(canvas, pts, fill=DEEP, edge=AZURE):
    """Filled tribal shape with a lit edge."""
    wrap_draw(canvas, pts, fill, fill=True)
    wrap_draw(canvas, pts, edge, thickness=1, closed=True)


def wrap_blur(img, k, sigma):
    """Blur with wraparound edges by blurring a 3x3 tiling and cropping."""
    big = np.tile(img, (3, 3, 1))
    big = cv2.GaussianBlur(big, (k, k), sigma)
    return big[SIZE:2 * SIZE, SIZE:2 * SIZE]


def rot(pts, angle):
    a = np.radians(angle)
    c, s = np.cos(a), np.sin(a)
    p = np.asarray(pts, np.float32)
    return np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c], 1)


def place(pts, center, scale=1.0, angle=0.0):
    """Motifs are authored in math space (y up); this maps them to image space."""
    p = rot(np.asarray(pts, np.float32) * scale, angle)
    p[:, 1] *= -1
    return p + np.asarray(center, np.float32)


def bezier(p0, p1, p2, n=64):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.asarray(p, np.float32) for p in (p0, p1, p2))
    return ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2).astype(np.float32)


def cubic(p0, p1, p2, p3, n=96):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float32) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3).astype(np.float32)


def taper(spine, w_base, w_tip=0.0, power=1.5):
    """Sweep a width profile along a curve to get a tapered ribbon that comes
    to a point. This is the basic tribal-tattoo stroke."""
    p = np.asarray(spine, np.float32)
    t = np.linspace(0, 1, len(p), dtype=np.float32)
    w = (w_base + (w_tip - w_base) * t ** power) / 2
    tang = np.gradient(p, axis=0)
    n = np.stack([-tang[:, 1], tang[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return np.concatenate([p + n * w[:, None], (p - n * w[:, None])[::-1]])


def blade(length, bend=0.7, width=15, n=110):
    """Curved claw/flame that tapers to a needle point."""
    spine = cubic((0, 0), (0.02 * length, 0.36 * length),
                  (0.34 * bend * length, 0.72 * length),
                  (0.78 * bend * length, length), n)
    return taper(spine, width, 0.0, 1.35)


def curl(R, turns=0.85, width=12, decay=0.62, n=150):
    """Inward spiral, tapering to a point at the center of the curl."""
    t = np.linspace(0, turns * 2 * np.pi, n)
    r = R * np.exp(-decay * t / (2 * np.pi))
    spine = np.stack([r * np.cos(t) - R, r * np.sin(t)], 1).astype(np.float32)
    return taper(spine, width, 0.0, 1.25)


def teardrop(R, m=2.6, n=240):
    """Classic teardrop: round bulb, cusp at the tip, pointing +y."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (np.stack([np.sin(t) * np.sin(t / 2) ** m, np.cos(t)], 1) * R).astype(np.float32)


def ogive(w, leg, n=64):
    """Equilateral pointed arch: two arcs of radius w meeting at the apex."""
    t = np.linspace(np.pi, 2 * np.pi / 3, n)
    left = np.stack([w / 2 + w * np.cos(t), w * np.sin(t)], 1)
    t = np.linspace(np.pi / 3, 0.0, n)
    right = np.stack([-w / 2 + w * np.cos(t), w * np.sin(t)], 1)
    return np.concatenate([[[-w / 2, -leg]], left, right, [[w / 2, -leg]]]).astype(np.float32)


def rosette(R, lobes=4, k=0.38, n=360, phase=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = R * (1 + k * np.cos(lobes * t + phase))
    return np.stack([r * np.cos(t), r * np.sin(t)], 1).astype(np.float32)


def offset_curve(pts, d):
    p = np.asarray(pts, np.float32)
    t = np.gradient(p, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return p + n * d


def band(canvas, pts, width, closed=False):
    """Two parallel rails, like carved stone tracery rather than a single line."""
    for d in (-width / 2, width / 2):
        stroke(canvas, offset_curve(pts, d), closed=closed)


def heading(d):
    """Angle that makes a +y motif point along image-space direction d."""
    d = np.asarray(d, np.float32)
    d = d / np.linalg.norm(d)
    return float(np.degrees(np.arctan2(-d[0], -d[1]))), np.array([-d[1], d[0]], np.float32)


def lattice(canvas, center, radius, bulge):
    """Diamond with ogee (outward-bowed) edges."""
    verts = [(0, radius), (radius, 0), (0, -radius), (-radius, 0)]
    for i in range(4):
        p0 = np.array(verts[i], np.float32)
        p2 = np.array(verts[(i + 1) % 4], np.float32)
        mid = (p0 + p2) / 2
        ctrl = mid + mid / np.linalg.norm(mid) * bulge
        curve = place(bezier(p0, ctrl, p2, n=160), center)
        band(canvas, curve, 11)
        wrap_draw(canvas, curve, WASH, thickness=1, closed=False)


def arcade(canvas, center, out_dir, count=3, spacing=23, w=18, leg=15):
    """A row of pointed arches standing on a lattice bar."""
    angle, tangent = heading(out_dir)
    d = np.asarray(out_dir, np.float32)
    d = d / np.linalg.norm(d)
    for i in range(count):
        pos = np.asarray(center, np.float32) + tangent * (i - (count - 1) / 2) * spacing + d * 5
        stroke(canvas, place(ogive(w, leg), pos, angle=angle), closed=False)


def medallion(canvas, center):
    for a in range(0, 360, 45):
        arch = ogive(38, 26)
        arch[:, 1] += 4
        stroke(canvas, place(arch, center + rot([[0, 118]], a)[0] * (1, -1), angle=a), closed=False)
    band(canvas, place(rosette(90, lobes=4, k=0.30), center), 9, closed=True)

    for a in range(0, 360, 45):
        drop = teardrop(34)
        drop[:, 1] += 42
        pts = place(drop, center, angle=a)
        solid(canvas, pts, fill=DEEP, edge=VIOLET)
        stroke(canvas, place(teardrop(19) + (0, 46), center, angle=a), closed=True,
               layers=((AZURE, 1),))

    stroke(canvas, place(rosette(20, lobes=4, k=0.42), center), closed=True)
    solid(canvas, place(rosette(10, lobes=4, k=0.5), center), fill=AZURE, edge=GLINT)


def tribal_node(canvas, center):
    """Four mirrored pairs of curls whipping out of a central teardrop."""
    for a in range(0, 360, 90):
        for s in (1, -1):
            spiral = curl(30, turns=0.9, width=12) * (s, 1) + (0, 12)
            solid(canvas, place(spiral, center, angle=a))
    for a in range(45, 360, 90):
        drop = teardrop(13) + (0, 16)
        solid(canvas, place(drop, center, angle=a), fill=VIOLET, edge=GLINT)
    solid(canvas, place(rosette(9, lobes=4, k=0.45), center), fill=AZURE, edge=GLINT)


def bar_flourish(canvas, center, out_dir):
    """Arches on the bar, flanked by blades that flick along it."""
    angle, tangent = heading(out_dir)
    arcade(canvas, center, out_dir)
    arcade(canvas, center, -np.asarray(out_dir, np.float32))
    for s in (1, -1):
        for t_dir in (tangent, -tangent):
            edge_angle, _ = heading(t_dir)
            flick = blade(44, bend=0.85, width=12) * (s, 1) + (0, 26)
            solid(canvas, place(flick, center, angle=edge_angle))


def build():
    # periodic wash: faint indigo bloom at the medallions, black between them
    u = np.linspace(0, 2 * np.pi, SIZE, endpoint=False)
    X, Y = np.meshgrid(u, u)
    field = (0.5 + 0.5 * np.cos(2 * X) * np.cos(2 * Y))[:, :, None]
    canvas = (np.array(BG, np.float32) + np.array(WASH, np.float32) * field ** 2 * 0.5).astype(np.uint8)

    ink = np.zeros((SIZE, SIZE, 3), np.uint8)

    # fine diagonal hatching (spacing divides SIZE, so it wraps cleanly)
    for c in range(-SIZE, SIZE * 2, 16):
        wrap_draw(ink, [(c, 0), (c + SIZE, SIZE)], HATCH, 1, closed=False)
        wrap_draw(ink, [(c, SIZE), (c + SIZE, 0)], HATCH, 1, closed=False)

    centers = [(HALF, HALF), (0, 0)]
    for c in centers:
        lattice(ink, c, HALF, 18)
    for m in [(128, 128), (384, 128), (128, 384), (384, 384)]:
        bar_flourish(ink, m, np.asarray(m, np.float32) - (HALF, HALF))
    for c in centers:
        medallion(ink, c)
    for v in [(HALF, 0), (0, HALF)]:
        tribal_node(ink, v)

    glow = wrap_blur(ink, 31, 9).astype(np.float32) * 0.45
    out = canvas.astype(np.float32) + glow + ink.astype(np.float32)

    rng = np.random.default_rng(11)
    out += rng.normal(0, 4.0, (SIZE, SIZE, 1))

    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    cv2.imwrite(OUT, build())
    print(f"wrote {OUT}")

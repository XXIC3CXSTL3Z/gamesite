"""Generates a scanned-paper texture into assets/paper.png."""

import os

import cv2
import numpy as np

W, H = 1240, 1754  # A4 at 150dpi
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "paper.png")


def octave(rng, scale, sigma=0.0):
    """Value noise: small random field scaled up, so it mottles at one frequency."""
    small = rng.normal(0, 1, (max(2, H // scale), max(2, W // scale)))
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(big, (0, 0), sigma) if sigma else big


def build():
    rng = np.random.default_rng(3)
    paper = np.full((H, W), 248.0)

    # fractal mottling: paper pulp is blotchy at every scale
    for scale, amp in [(1, 2.6), (3, 2.2), (9, 2.4), (28, 2.6), (90, 2.8), (240, 3.0)]:
        paper += octave(rng, scale) * amp

    # fibres: noise smeared along random directions
    for angle in (12, 78, 143):
        line = np.zeros((21, 21), np.float32)
        line[10, :] = 1.0
        line = cv2.warpAffine(line, cv2.getRotationMatrix2D((10, 10), angle, 1.0), (21, 21))
        paper += cv2.filter2D(octave(rng, 2), -1, line / line.sum()) * 2.2

    # flecks and specks embedded in the pulp
    for _ in range(900):
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.circle(paper, (int(x), int(y)), int(rng.integers(1, 3)),
                   float(rng.uniform(190, 225)), -1, cv2.LINE_AA)

    # a soft horizontal fold and a vertical one, as if it had been folded in quarters
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    paper -= 9.0 * np.exp(-((ys - H * 0.5) ** 2) / (2 * 5.0 ** 2))
    paper -= 6.0 * np.exp(-((xs - W * 0.5) ** 2) / (2 * 4.0 ** 2))
    paper += 4.0 * np.exp(-((ys - H * 0.5 - 7) ** 2) / (2 * 7.0 ** 2))

    # scanner lamp falloff: edges sit slightly darker than the middle
    gx = (xs / W - 0.5) * 2
    gy = (ys / H - 0.5) * 2
    paper -= (gx ** 2 * 6.0 + gy ** 2 * 9.0)

    # faint vertical banding from the sensor
    paper += np.sin(xs / 3.1) * 0.6 + octave(rng, 400) * 1.2

    paper = np.clip(paper, 0, 255)

    # paper is warm, and scans push a little cyan into the shadows
    out = np.stack([paper * 0.972, paper * 0.988, paper], axis=2)
    out += rng.normal(0, 1.6, (H, W, 3))
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    cv2.imwrite(OUT, build())
    print(f"wrote {OUT}")

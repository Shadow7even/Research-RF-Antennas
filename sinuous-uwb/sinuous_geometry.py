"""
4-arm sinuous antenna geometry (pure Python, no Ansys needed).

Run this file on its own to get a preview picture of the antenna:
    python sinuous_geometry.py
It writes sinuous_preview.png next to this file.

THE DESIGN EQUATIONS (DuHamel sinuous antenna)
    Center line of one arm, in polar coordinates (r, phi):
        phi(r) = alpha * sin( pi * ln(r / R_out) / ln(tau) )
    The arm is the strip between phi(r) - delta and phi(r) + delta.
    The 4 arms are copies rotated by 0, 90, 180 and 270 degrees.

    alpha : how far the arm swings left/right (angle)
    delta : half the angular width of the arm. delta = 22.5 deg with 4 arms makes the
            antenna "self-complementary" (metal and gaps have the same shape), which is what
            keeps the impedance almost constant over a very wide band.
    tau   : growth ratio between neighbouring zig-zag cells (0 < tau < 1).

    A cell radiates when it is about half a wavelength long, which gives the rule
        r  ~  lambda / (4 * (alpha + delta))        (angles in radians)
    so the OUTER radius sets the LOWEST frequency and the INNER radius sets the HIGHEST.

FEED REGION
    Inside the sinuous part the arms continue as straight 45-degree sectors down to a small
    feed radius, where four small lumped ports connect each arm to a tiny centre pad.
    Opposite arms (1 & 3, 2 & 4) are driven 180 degrees apart in post-processing, which is
    exactly what a balun would do in the real antenna.
"""

import math
from dataclasses import dataclass, field

import numpy as np

C0 = 299792458.0


@dataclass
class SinuousParams:
    f_low_ghz: float = 2.0          # lowest design frequency
    f_high_ghz: float = 10.0        # highest design frequency
    alpha_deg: float = 45.0         # arm swing angle
    delta_deg: float = 22.5         # arm half-width (22.5 = self-complementary for 4 arms)
    tau: float = 0.8                # cell growth ratio
    low_margin: float = 1.10        # outer radius made 10% larger than the bare rule
    high_margin: float = 0.80       # inner radius made 20% smaller than the bare rule
    r_feed_mm: float = 2.0          # where the arms end and the ports start
    r_pad_mm: float = 0.6           # radius of the small centre pad
    points_per_cell: int = 30       # curve resolution (more = smoother, slower to mesh)
    n_arms: int = 4

    # derived values (filled in by __post_init__)
    r_out_mm: float = field(init=False)
    r_in_mm: float = field(init=False)
    n_cells: float = field(init=False)

    def __post_init__(self):
        k = math.radians(self.alpha_deg + self.delta_deg)
        lam_low = C0 / (self.f_low_ghz * 1e9) * 1e3
        lam_high = C0 / (self.f_high_ghz * 1e9) * 1e3
        self.r_out_mm = self.low_margin * lam_low / (4 * k)
        self.r_in_mm = self.high_margin * lam_high / (4 * k)
        if not (0 < self.tau < 1):
            raise ValueError("tau must be between 0 and 1")
        if not (self.r_pad_mm < self.r_feed_mm < self.r_in_mm < self.r_out_mm):
            raise ValueError(
                f"Radii must satisfy r_pad < r_feed < r_in < r_out, got "
                f"{self.r_pad_mm:.2f} < {self.r_feed_mm:.2f} < {self.r_in_mm:.2f} < {self.r_out_mm:.2f}")
        self.n_cells = math.log(self.r_out_mm / self.r_in_mm) / math.log(1 / self.tau)

    def summary(self) -> str:
        return (f"Band {self.f_low_ghz:g}-{self.f_high_ghz:g} GHz | "
                f"outer radius {self.r_out_mm:.1f} mm (diameter {2 * self.r_out_mm:.1f} mm) | "
                f"inner radius {self.r_in_mm:.2f} mm | {self.n_cells:.1f} cells | "
                f"alpha={self.alpha_deg:g} deg, delta={self.delta_deg:g} deg, tau={self.tau:g}")


def _center_angle(p: SinuousParams, r: np.ndarray) -> np.ndarray:
    """Arm center-line angle (radians) for radius r (mm)."""
    alpha = math.radians(p.alpha_deg)
    r = np.asarray(r, dtype=float)
    phi = alpha * np.sin(math.pi * np.log(r / p.r_out_mm) / math.log(p.tau))
    # straight sector inside the sinuous part (feed region)
    phi_in = alpha * math.sin(math.pi * math.log(p.r_in_mm / p.r_out_mm) / math.log(p.tau))
    return np.where(r < p.r_in_mm, phi_in, phi)


def _radii(p: SinuousParams) -> np.ndarray:
    """Sample radii from feed to outer edge, uniform in log(r) inside the sinuous part."""
    n_sin = max(int(math.ceil(p.n_cells * p.points_per_cell)), 20)
    r_sin = np.exp(np.linspace(math.log(p.r_in_mm), math.log(p.r_out_mm), n_sin))
    r_feed = np.linspace(p.r_feed_mm, p.r_in_mm, 4, endpoint=False)
    return np.concatenate([r_feed, r_sin])


def feed_angle(p: SinuousParams) -> float:
    """Center angle of arm 0 in the feed region (radians)."""
    return float(_center_angle(p, np.array([p.r_feed_mm]))[0])


def arm_polygons(p: SinuousParams) -> list:
    """Return a list of n_arms polygons; each polygon is an (N, 2) array of x, y in mm.
    Point order: out along the +delta edge, back along the -delta edge.
    First point = (r_feed, +delta), last point = (r_feed, -delta), so the closing segment
    is the straight 'tip' that the port touches."""
    r = _radii(p)
    phi = _center_angle(p, r)
    d = math.radians(p.delta_deg)
    polys = []
    for k in range(p.n_arms):
        rot = 2 * math.pi * k / p.n_arms
        a_plus = phi + d + rot
        a_minus = (phi - d + rot)[::-1]
        # round outer end: a straight chord there would cut through the wiggling arm edges
        a_cap = np.linspace(phi[-1] + d, phi[-1] - d, 9)[1:-1] + rot
        rr = np.concatenate([r, np.full(a_cap.size, r[-1]), r[::-1]])
        aa = np.concatenate([a_plus, a_cap, a_minus])
        polys.append(np.column_stack([rr * np.cos(aa), rr * np.sin(aa)]))
    return polys


def port_quads(p: SinuousParams) -> list:
    """One quadrilateral per arm joining the arm tip (r_feed) to the centre pad (r_pad).
    Returns list of dicts: points (4x2), line_start (pad side), line_stop (arm side)."""
    d = math.radians(p.delta_deg)
    th0 = feed_angle(p)
    arms = arm_polygons(p)
    quads = []
    for k in range(p.n_arms):
        th = th0 + 2 * math.pi * k / p.n_arms
        a1, a2 = th - d, th + d
        # reuse the arm's own tip corners so port and arm share the edge exactly
        tip_minus, tip_plus = arms[k][-1], arms[k][0]
        pts = np.array([
            [p.r_pad_mm * math.cos(a1), p.r_pad_mm * math.sin(a1)],
            tip_minus,
            tip_plus,
            [p.r_pad_mm * math.cos(a2), p.r_pad_mm * math.sin(a2)],
        ])
        # integration line: from the middle of the pad edge to the middle of the arm tip
        start = 0.5 * (pts[0] + pts[3])
        stop = 0.5 * (pts[1] + pts[2])
        quads.append({"points": pts, "line_start": start, "line_stop": stop})
    return quads


def pad_polygon(p: SinuousParams) -> np.ndarray:
    """Centre pad: polygon through the inner corners of all port quads (a regular octagon
    for the 4 arms), built from the very same coordinates so every port shares an edge with
    the pad exactly (no floating-point gaps)."""
    pts = []
    for q in port_quads(p):
        pts += [q["points"][0], q["points"][3]]
    pts = np.array(pts)
    order = np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))
    return pts[order]


def save_preview(p: SinuousParams, path: str, title_extra: str = "") -> str:
    """Draw the arms, ports and pad to a PNG (used for the report and for a quick sanity check)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle

    colors = ["#c0392b", "#2471a3", "#c0392b", "#2471a3"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))
    for ax, zoom in zip(axes, (p.r_out_mm * 1.08, p.r_in_mm * 1.6)):
        for k, poly in enumerate(arm_polygons(p)):
            ax.add_patch(Polygon(poly, closed=True, fc=colors[k % 4], ec="black", lw=0.3, alpha=0.85))
        for k, q in enumerate(port_quads(p)):
            ax.add_patch(Polygon(q["points"], closed=True, fc="gold", ec="black", lw=0.4))
            c = q["points"].mean(axis=0)
            if zoom < p.r_out_mm:
                ax.annotate(f"P{k + 1}", c * 1.9, ha="center", va="center", fontsize=9, weight="bold")
        ax.add_patch(Polygon(pad_polygon(p), closed=True, fc="gray", ec="black", lw=0.4))
        ax.add_patch(Circle((0, 0), p.r_out_mm, fill=False, ls="--", lw=0.6, color="gray"))
        ax.add_patch(Circle((0, 0), p.r_in_mm, fill=False, ls=":", lw=0.6, color="gray"))
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(-zoom, zoom)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    axes[0].set_title(f"Full antenna, diameter {2 * p.r_out_mm:.0f} mm")
    axes[1].set_title("Feed region: ports P1-P4 (gold) and centre pad (gray)")
    fig.suptitle(p.summary() + (("\n" + title_extra) if title_extra else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    import os
    params = SinuousParams()
    print(params.summary())
    out = save_preview(params, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sinuous_preview.png"))
    print("Preview written to", out)

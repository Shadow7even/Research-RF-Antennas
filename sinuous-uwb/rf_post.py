"""
Post-processing for the sinuous antenna (pure Python: numpy + matplotlib, no Ansys needed).

run_sinuous.py saves the raw simulation data into a results folder, then calls this file.
You can also re-make every plot later, on any computer, from the saved data:
    python rf_post.py results/<run_folder>

WHAT IS COMPUTED
  The model has 4 single-ended ports (P1..P4), one per arm. A real sinuous antenna is fed
  in opposite pairs through a balun, so we combine the ports in "mixed-mode" form:
    Polarization X = arms 1 & 3 driven 180 deg apart (ports P1, P3)
    Polarization Y = arms 2 & 4 driven 180 deg apart (ports P2, P4)
  Sdd11 = (S11 - S13 - S31 + S33) / 2      -> how well polarization X is matched
  Sdd21 = (S21 - S23 - S41 + S43) / 2      -> leakage from X into Y (isolation)
  Reference impedance of a pair = 2 x port impedance (2 x 66.5 = 133 ohm by default),
  which is what the Klopfenstein balun will transform to 50 ohm in the full antenna.
"""

import csv
import math
import os
import re
import sys

import numpy as np

# Fixed categorical colours (always in this order, never cycled)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#d9d9d9"


# --------------------------------------------------------------------------------------
# S-parameter math
# --------------------------------------------------------------------------------------
def read_touchstone(path):
    """Read an N-port Touchstone v1 file. Returns (freq_hz, S[F,N,N], z0, port_names)."""
    n_ports = int(re.search(r"\.s(\d+)p$", path, re.I).group(1))
    unit_scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
    scale, fmt, z0 = 1e9, "MA", 50.0
    names = {}
    numbers = []
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("!"):
                m = re.search(r"Port\[(\d+)\]\s*=\s*([^\s]+)", line)
                if m:
                    names[int(m.group(1))] = m.group(2).split(":")[0]
                continue
            if line.startswith("#"):
                tok = line[1:].upper().split()
                for i, t in enumerate(tok):
                    if t in unit_scale:
                        scale = unit_scale[t]
                    elif t in ("MA", "DB", "RI"):
                        fmt = t
                    elif t == "R" and i + 1 < len(tok):
                        z0 = float(tok[i + 1])
                continue
            line = line.split("!")[0]
            numbers.extend(float(x) for x in line.split())
    per_freq = 1 + 2 * n_ports * n_ports
    data = np.array(numbers)
    if data.size % per_freq != 0:
        raise ValueError(f"Touchstone data length {data.size} is not a multiple of {per_freq}")
    data = data.reshape(-1, per_freq)
    freq = data[:, 0] * scale
    a, b = data[:, 1::2], data[:, 2::2]
    if fmt == "RI":
        s = a + 1j * b
    elif fmt == "MA":
        s = a * np.exp(1j * np.radians(b))
    else:  # DB
        s = 10 ** (a / 20) * np.exp(1j * np.radians(b))
    s = s.reshape(-1, n_ports, n_ports)
    if n_ports == 2:  # 2-port files are stored S11 S21 S12 S22
        s = s.transpose(0, 2, 1)
    port_names = [names.get(i + 1, f"P{i + 1}") for i in range(n_ports)]
    return freq, s, z0, port_names


def renormalize(s, z_old, z_new):
    """Change the reference impedance of all ports from z_old to z_new (both real, ohms)."""
    if abs(z_old - z_new) < 1e-9:
        return s.copy()
    # Formula without the Z-matrix (Z can be singular for floating arms):
    #   S' = (S - g I)(I - g S)^-1,  g = (z_new - z_old) / (z_new + z_old)
    n = s.shape[-1]
    eye = np.eye(n)
    g = (z_new - z_old) / (z_new + z_old)
    return (s - g * eye) @ np.linalg.inv(eye - g * s)


def mixed_mode(s, pair_a=(0, 2), pair_b=(1, 3)):
    """Differential-mode S for two port pairs. Returns sdd_aa, sdd_bb, sdd_ba."""
    (i, j), (k, l) = pair_a, pair_b
    sdd_aa = (s[:, i, i] - s[:, i, j] - s[:, j, i] + s[:, j, j]) / 2
    sdd_bb = (s[:, k, k] - s[:, k, l] - s[:, l, k] + s[:, l, l]) / 2
    sdd_ba = (s[:, k, i] - s[:, k, j] - s[:, l, i] + s[:, l, j]) / 2
    return sdd_aa, sdd_bb, sdd_ba


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def vswr(gamma):
    g = np.minimum(np.abs(gamma), 0.999999)
    return (1 + g) / (1 - g)


def bands_below(freq_ghz, values_db, limit_db):
    """Contiguous frequency ranges where values_db <= limit_db. Returns list of (f1, f2)."""
    ok = values_db <= limit_db
    out, start = [], None
    for f, flag in zip(freq_ghz, ok):
        if flag and start is None:
            start = f
        if not flag and start is not None:
            out.append((start, prev))
            start = None
        prev = f
    if start is not None:
        out.append((start, freq_ghz[-1]))
    return out


# --------------------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "legend.frameon": False, "lines.linewidth": 2,
    })
    return plt


def _band_shade(ax, f_low, f_high):
    ax.axvspan(f_low, f_high, color="#2a78d6", alpha=0.06, lw=0)
    ax.text(f_low, 0.97, f" design band {f_low:g}-{f_high:g} GHz", transform=ax.get_xaxis_transform(),
            color=MUTED, fontsize=8, va="top")


def plot_sparams(out_dir, f_ghz, sdd11, sdd22, sdd21, zref, f_low, f_high):
    plt = _plt()
    paths = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _band_shade(ax, f_low, f_high)
    ax.plot(f_ghz, db(sdd11), color=C1, label="Pol X  |Sdd11| (ports 1 & 3)")
    ax.plot(f_ghz, db(sdd22), color=C2, label="Pol Y  |Sdd22| (ports 2 & 4)", ls="--")
    ax.axhline(-10, color=MUTED, lw=1, ls=":")
    ax.text(f_ghz[0], -10, " -10 dB", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Return loss (dB)")
    ax.set_title(f"Input match of each polarization (reference {zref:g} ohm differential)", color=INK)
    ax.set_ylim(min(-40, np.nanmin(db(sdd11)) - 2), 0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    paths.append(os.path.join(out_dir, "01_return_loss.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _band_shade(ax, f_low, f_high)
    ax.plot(f_ghz, db(sdd21), color=C3, label="|Sdd21|  Pol X -> Pol Y")
    ax.axhline(-20, color=MUTED, lw=1, ls=":")
    ax.text(f_ghz[0], -20, " -20 dB", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Coupling (dB)")
    ax.set_title("Isolation between the two polarizations (lower is better)", color=INK)
    ax.legend(loc="upper right")
    fig.tight_layout()
    paths.append(os.path.join(out_dir, "02_isolation.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _band_shade(ax, f_low, f_high)
    ax.plot(f_ghz, np.minimum(vswr(sdd11), 10), color=C1, label="Pol X")
    ax.axhline(2, color=MUTED, lw=1, ls=":")
    ax.text(f_ghz[0], 2, " VSWR 2", color=MUTED, fontsize=8, va="bottom")
    ax.set_ylim(1, 6)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("VSWR")
    ax.set_title("VSWR (values above 6 clipped)", color=INK)
    fig.tight_layout()
    paths.append(os.path.join(out_dir, "03_vswr.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    zin = zref * (1 + sdd11) / (1 - sdd11)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax in axes:
        _band_shade(ax, f_low, f_high)
    axes[0].plot(f_ghz, zin.real, color=C1)
    axes[0].axhline(zref, color=MUTED, lw=1, ls=":")
    axes[0].set_ylabel("Resistance (ohm)")
    axes[0].set_title(f"Differential input impedance, Pol X (target {zref:g} ohm, 0 reactance)", color=INK)
    axes[1].plot(f_ghz, zin.imag, color=C1)
    axes[1].axhline(0, color=MUTED, lw=1, ls=":")
    axes[1].set_ylabel("Reactance (ohm)")
    axes[1].set_xlabel("Frequency (GHz)")
    lim = max(50, min(400, np.nanpercentile(np.abs(zin.imag), 98) * 1.1))
    axes[1].set_ylim(-lim, lim)
    axes[0].set_ylim(0, max(2.5 * zref, 1))
    fig.tight_layout()
    paths.append(os.path.join(out_dir, "04_impedance.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    paths.append(plot_smith(out_dir, f_ghz, sdd11, zref, f_low, f_high))
    return paths


def plot_smith(out_dir, f_ghz, gamma, zref, f_low, f_high):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(False)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1)
    for r in (0.2, 0.5, 1, 2, 5):
        c, rad = r / (1 + r), 1 / (1 + r)
        ax.plot(c + rad * np.cos(t), rad * np.sin(t), color=GRID, lw=0.7)
    for x in (0.2, 0.5, 1, 2, 5):
        for sgn in (1, -1):
            cx, cy, rad = 1, sgn / x, 1 / x
            xs, ys = cx + rad * np.cos(t), cy + rad * np.sin(t)
            inside = np.hypot(xs, ys) <= 1.0001
            ax.plot(np.where(inside, xs, np.nan), np.where(inside, ys, np.nan), color=GRID, lw=0.7)
    ax.axhline(0, color=GRID, lw=0.7)
    in_band = (f_ghz >= f_low) & (f_ghz <= f_high)
    ax.plot(gamma.real, gamma.imag, color="#a8a8a8", lw=1.2, label="outside design band")
    ax.plot(np.where(in_band, gamma.real, np.nan), np.where(in_band, gamma.imag, np.nan),
            color=C1, lw=2, label="inside design band")
    vs2 = 1 / 3  # |gamma| for VSWR = 2
    ax.plot(vs2 * np.cos(t), vs2 * np.sin(t), color=MUTED, lw=1, ls=":", label="VSWR = 2 circle")
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_title(f"Smith chart, Pol X (normalized to {zref:g} ohm)", color=INK)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "05_smith_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Far-field data: long CSV with columns mode, freq_ghz, phi_deg, theta_deg, quantity, value_db
# --------------------------------------------------------------------------------------
def read_patterns(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["mode"], float(r["freq_ghz"]), float(r["phi_deg"]), float(r["theta_deg"]),
                         r["quantity"], float(r["value_db"])))
    return rows


def _series(rows, mode, freq, phi, quantity):
    pts = sorted((t, v) for m, f, p, t, q, v in rows
                 if m == mode and abs(f - freq) < 1e-6 and abs(p - phi) < 1e-6 and q == quantity)
    if not pts:
        return np.array([]), np.array([])
    a = np.array(pts)
    return a[:, 0], a[:, 1]


def boresight(rows, mode, quantity):
    """Value at theta = 0, phi = 0 for every frequency available."""
    freqs = sorted({f for m, f, p, t, q, v in rows if m == mode and q == quantity})
    out = []
    for f in freqs:
        th, val = _series(rows, mode, f, 0.0, quantity)
        if th.size:
            out.append((f, float(np.interp(0.0, th, val))))
    return np.array(out) if out else np.zeros((0, 2))


def plot_patterns(out_dir, rows):
    plt = _plt()
    paths = []
    freqs = sorted({f for m, f, p, t, q, v in rows})
    if not freqs:
        return paths
    specs = [("X", [("RealizedGainTotal", 0.0, C1, "-", "phi = 0 (E-plane)"),
                    ("RealizedGainTotal", 90.0, C2, "--", "phi = 90 (H-plane)")],
              "Pol X (arms 1 & 3): realized gain, dBi", "06_patterns_polX.png"),
             ("CP", [("RealizedGainRHCP", 0.0, C1, "-", "RHCP, phi = 0"),
                     ("RealizedGainLHCP", 0.0, C2, "--", "LHCP, phi = 0")],
              "Circular mode (all 4 arms, 0/90/180/270 deg): realized gain, dBi", "07_patterns_CP.png")]
    for mode, curves, title, fname in specs:
        if not any(m == mode for m, *_ in rows):
            continue
        n = len(freqs)
        cols = min(4, n)
        nrows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(nrows, cols, figsize=(3.4 * cols, 4.1 * nrows),
                                 subplot_kw={"projection": "polar"}, squeeze=False)
        peak = max((v for m, f, p, t, q, v in rows if m == mode), default=0)
        top = 5 * math.ceil((peak + 1) / 5)
        for ax, f in zip(axes.flat, freqs):
            for qty, phi, col, ls, lab in curves:
                th, val = _series(rows, mode, f, phi, qty)
                if th.size:
                    ax.plot(np.radians(th), np.clip(val, top - 40, None), color=col, ls=ls, lw=1.6, label=lab)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(top - 40, top)
            ax.set_yticks([top - 30, top - 20, top - 10, top])
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.set_title(f"{f:g} GHz", color=INK, fontsize=10, pad=10)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
        fig.suptitle(title + "  (0 deg = straight up from the antenna)", color=INK)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=2.5)
        paths.append(os.path.join(out_dir, fname))
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)

    # gain and axial ratio at boresight vs frequency (two separate charts, one axis each)
    gx = boresight(rows, "X", "RealizedGainTotal")
    gr = boresight(rows, "CP", "RealizedGainRHCP")
    gl = boresight(rows, "CP", "RealizedGainLHCP")
    if gx.size or gr.size:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if gx.size:
            ax.plot(gx[:, 0], gx[:, 1], "-o", color=C1, ms=6, label="Pol X, total")
        if gr.size:
            ax.plot(gr[:, 0], gr[:, 1], "-o", color=C2, ms=6, label="Circular mode, RHCP")
        if gl.size:
            ax.plot(gl[:, 0], gl[:, 1], "--o", color=C3, ms=6, label="Circular mode, LHCP")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Realized gain at boresight (dBi)")
        ax.set_title("Boresight gain vs frequency (theta = 0)", color=INK)
        ax.legend(loc="best")
        fig.tight_layout()
        paths.append(os.path.join(out_dir, "08_gain_vs_freq.png"))
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    ar = boresight(rows, "CP", "AxialRatio")
    if ar.size:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(ar[:, 0], ar[:, 1], "-o", color=C1, ms=6)
        ax.axhline(3, color=MUTED, lw=1, ls=":")
        ax.text(ar[0, 0], 3, " 3 dB", color=MUTED, fontsize=8, va="bottom")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Axial ratio (dB)")
        ax.set_title("Axial ratio at boresight, circular mode (below 3 dB = good CP)", color=INK)
        ax.set_ylim(0, max(10, float(np.nanmax(ar[:, 1])) + 1))
        fig.tight_layout()
        paths.append(os.path.join(out_dir, "09_axial_ratio.png"))
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    return paths


# --------------------------------------------------------------------------------------
# Whole run
# --------------------------------------------------------------------------------------
def process_run(run_dir, f_low=None, f_high=None, z_port=None):
    """Make every plot + summary.md from the files saved in run_dir."""
    meta = {}
    meta_path = os.path.join(run_dir, "run_info.csv")
    if os.path.exists(meta_path):
        with open(meta_path, newline="") as fh:
            meta = {r["key"]: r["value"] for r in csv.DictReader(fh)}
    f_low = f_low or float(meta.get("f_low_ghz", 0.8))
    f_high = f_high or float(meta.get("f_high_ghz", 8.0))
    z_port = z_port or float(meta.get("port_impedance_ohm", 66.5))
    zref = 2 * z_port

    plots, lines = [], []
    lines.append(f"# Sinuous antenna results - {meta.get('run_name', os.path.basename(run_dir.rstrip('/')))}\n")
    if meta:
        lines.append("| Setting | Value |\n|---|---|")
        for k in ("mode", "f_low_ghz", "f_high_ghz", "outer_diameter_mm", "inner_radius_mm", "n_cells",
                  "alpha_deg", "delta_deg", "tau", "substrate", "substrate_thickness_mm", "port_impedance_ohm",
                  "adapt_freq_ghz", "max_passes", "solve_time"):
            if k in meta:
                lines.append(f"| {k} | {meta[k]} |")
        lines.append("")
    if os.path.exists(os.path.join(run_dir, "geometry_preview.png")):
        lines.append("## Geometry\n\n![geometry](geometry_preview.png)\n")

    npz = os.path.join(run_dir, "sparams_raw.npz")
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=False)
        freq, s, z0 = d["freq_hz"], d["s"], float(d["z0"])
        s = renormalize(s, z0, z_port)
        f_ghz = freq / 1e9
        sdd11, sdd22, sdd21 = mixed_mode(s)
        with open(os.path.join(run_dir, "sparams_mixed_mode.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["freq_ghz", "sdd11_db", "sdd22_db", "sdd21_db", "vswr_x", "zin_re_ohm", "zin_im_ohm"])
            zin = zref * (1 + sdd11) / (1 - sdd11)
            for i in range(f_ghz.size):
                w.writerow([f"{f_ghz[i]:.5f}", f"{db(sdd11[i]):.3f}", f"{db(sdd22[i]):.3f}",
                            f"{db(sdd21[i]):.3f}", f"{vswr(sdd11[i]):.3f}", f"{zin[i].real:.2f}",
                            f"{zin[i].imag:.2f}"])
        plots += plot_sparams(run_dir, f_ghz, sdd11, sdd22, sdd21, zref, f_low, f_high)

        inb = (f_ghz >= f_low) & (f_ghz <= f_high)
        rl = db(sdd11)
        good = bands_below(f_ghz, rl, -10)
        frac = float(np.mean(rl[inb] <= -10)) * 100 if inb.any() else float("nan")
        lines.append("## Key numbers\n")
        lines.append(f"- Reference impedance: {zref:g} ohm differential per polarization "
                     f"(= what the balun must match to 50 ohm).")
        lines.append(f"- Frequency ranges with |Sdd11| below -10 dB: " +
                     (", ".join(f"{a:.2f}-{b:.2f} GHz" for a, b in good) if good else "none"))
        lines.append(f"- Share of the design band ({f_low:g}-{f_high:g} GHz) with |Sdd11| below -10 dB: {frac:.0f}%")
        if inb.any():
            lines.append(f"- Worst |Sdd11| inside the band: {rl[inb].max():.1f} dB "
                         f"at {f_ghz[inb][np.argmax(rl[inb])]:.2f} GHz")
            iso = db(sdd21)[inb]
            lines.append(f"- Worst polarization isolation inside the band: {iso.max():.1f} dB")
            zr = (zref * (1 + sdd11) / (1 - sdd11)).real[inb]
            lines.append(f"- Input resistance inside the band: {zr.min():.0f} to {zr.max():.0f} ohm "
                         f"(median {np.median(zr):.0f} ohm)")
        lines.append("")
    else:
        lines.append("_No S-parameter data found (sparams_raw.npz missing)._\n")

    pat = os.path.join(run_dir, "patterns.csv")
    if os.path.exists(pat):
        rows = read_patterns(pat)
        plots += plot_patterns(run_dir, rows)
        gx = boresight(rows, "X", "RealizedGainTotal")
        ar = boresight(rows, "CP", "AxialRatio")
        if gx.size:
            lines.append("- Boresight realized gain, Pol X: " +
                         ", ".join(f"{f:g} GHz: {g:.1f} dBi" for f, g in gx))
        if ar.size:
            lines.append("- Boresight axial ratio, circular mode: " +
                         ", ".join(f"{f:g} GHz: {a:.1f} dB" for f, a in ar))
        lines.append("")

    lines.append("## Plots\n")
    for p in plots:
        lines.append(f"![{os.path.basename(p)}]({os.path.basename(p)})\n")
    field_imgs = sorted(x for x in os.listdir(run_dir) if x.startswith("field_") and x.endswith((".png", ".jpg")))
    if field_imgs:
        lines.append("## Surface current on the arms\n")
        for x in field_imgs:
            lines.append(f"![{x}]({x})\n")
    lines.append("## How to read this\n")
    lines.append("- **Return loss / VSWR**: below -10 dB (VSWR below 2) means at least 90% of the power "
                 "goes into the antenna instead of bouncing back.")
    lines.append("- **Isolation**: how much of polarization X leaks into polarization Y. Below -20 dB is good.")
    lines.append("- **Impedance**: a self-complementary 4-arm sinuous should sit near 133 ohm with small "
                 "reactance across the band; big swings show where the geometry is truncated.")
    lines.append("- **Patterns**: this model has no cavity yet, so the antenna radiates up and down "
                 "equally (two lobes). The cavity + absorber step removes the downward lobe.")
    lines.append("- **Axial ratio**: below 3 dB means the circular mode really is circular.")
    with open(os.path.join(run_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return os.path.join(run_dir, "summary.md"), plots


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rf_post.py <results/run_folder>")
        sys.exit(1)
    summary, made = process_run(sys.argv[1])
    print(f"Made {len(made)} plots. Summary: {summary}")

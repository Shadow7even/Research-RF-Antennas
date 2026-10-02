"""
80 GHz series-fed patch array (transmission-line / leaky-wave style) - HFSS + PyAEDT
Revised version of the script shared as "patch array(1).txt".

WHAT THE SCRIPT DOES
  - Builds a 22 mm board: ground plane (z = -h/2), dielectric slab, and a chain of
    [strip - patch - strip - patch ... - strip] on top (z = +h/2), plus a tapered feed
    (trapezoid + rectangle) and a lumped port at each end of the board.
  - Solves at 80 GHz, reads S11, S21 and the broadside gain pattern, combines them
    into one "error" number, and lets SciPy change the dimensions to minimize it.

MAIN CHANGES vs. the original (see the chat for the reasoning)
  1. Targets AEDT 2025.2 (VM has v252) instead of 2023.1; removed the hard-coded v231 path.
  2. Launches ONE AEDT session (the original opened two and used two licenses).
  3. Creates the far-field infinite sphere "Elevation" - the original read gain from a
     sphere that was never created, so the gain step could not work.
  4. Gain sign in the objective fixed: the optimizer now MAXIMIZES broadside gain
     (the original minimized it). Confirm this matches what your friend intended.
  5. Number of patches is fixed per run (NUM_PATCHES). A gradient optimizer cannot
     handle an integer parameter; loop over 2..5 patches as separate runs instead.
  6. Geometries that do not fit on the board (negative feed length) are rejected with a
     penalty instead of being sent to HFSS.
  7. Optimizer changed L-BFGS-B -> Nelder-Mead with a cap on evaluations (MAX_EVALS);
     L-BFGS-B needed n+1 full solves per step.
  8. Port names are read from HFSS instead of guessed ("..._T1" naming varies).
  9. Log file is written correctly (all parameters) and flushed each iteration, so a
     crash does not lose results. Duplicate object name on the last strip fixed.
 10. Debug modes: MODE = "geometry" (build + validate, no solve), "single" (one solve),
     "optimize". SMOKE_TEST = True uses a coarse mesh for fast checks.
 11. Plots are saved to PNG (no interactive window needed); 3D screenshot is optional.
 12. AEDT is released in a finally block so a crash does not hold the license.
"""

import os
import socket
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save figures to files; works over remote desktop and non-graphical mode
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import windows

try:  # PyAEDT >= 0.9 package name
    from ansys.aedt.core import Hfss
except ImportError:  # older PyAEDT
    from pyaedt import Hfss

###############################################################################
# RUN SETTINGS  <-- change these
###############################################################################
MODE = "geometry"        # "geometry" -> build + validate only (seconds)
                         # "single"   -> one full solve with the initial parameters
                         # "optimize" -> full optimization
SMOKE_TEST = True        # True = coarse mesh (few passes) for quick debugging
NUM_PATCHES = 5          # fixed per run; run again with 2, 3, 4 to compare
MAX_EVALS = 40           # max number of HFSS solves in "optimize" mode
AEDT_VERSION = "2025.2"
NON_GRAPHICAL = False    # False = you see the HFSS window on the remote desktop
KEEP_AEDT_OPEN = True    # True = leave AEDT open at the end so you can inspect the model

try:  # lab-specific helper file (not shared with the script); fall back if missing
    import uncc_mts_compute_config as compute_config
    _cfg = compute_config.SolverConfig().solver_config
    NUM_CORES, NUM_GPUS = _cfg["num_cores"], _cfg["num_gpu"]
except Exception:
    NUM_CORES, NUM_GPUS = 4, 0

###############################################################################
# DESIGN CONSTANTS
###############################################################################
C0 = 2.99792458e8
FREQ_GHZ = 80.0
FREQ_STR = f"{FREQ_GHZ:g}GHz"                      # "80GHz" - same string everywhere
WAVELENGTH_MM = 1e3 * C0 / (FREQ_GHZ * 1e9)        # ~3.75 mm

BOARD_LENGTH_MM = 22.0
BOARD_WIDTH_MM = 3.0
HEIGHT_MM = 0.09            # dielectric thickness
PORT_WIDTH_MM = 0.5         # lumped port width = wide end of the feed taper
MIN_FEED_MM = 0.05          # feeds shorter than this are treated as invalid geometry

DIELECTRIC_MATERIAL = "glass"
SETUP_NAME = "LW_TL_Antenna_Setup"
SPHERE_NAME = "Elevation"

METAL_COLOR = [143, 175, 143]
DIELECTRIC_COLOR = [255, 255, 128]
PORT_COLOR = [128, 255, 255]

# Optimized parameters (num_patches is fixed above, not optimized)
PARAM_NAMES = ["strip_width_mm", "strip_length_mm", "patch_width_mm",
               "patch_length_mm", "feed_trap_pct"]
INITIAL = np.array([0.2,                    # strip width
                    WAVELENGTH_MM / 2,      # strip length
                    WAVELENGTH_MM / 2,      # patch width
                    WAVELENGTH_MM / 2,      # patch length
                    0.5])                   # fraction of the feed that is tapered
BOUNDS = [(0.1, 0.24),
          (0.1 * WAVELENGTH_MM / 2, 1.2 * WAVELENGTH_MM / 2),
          (0.5 * WAVELENGTH_MM / 2, 1.2 * WAVELENGTH_MM / 2),
          (0.5 * WAVELENGTH_MM / 2, 1.2 * WAVELENGTH_MM / 2),
          (0.1, 0.9)]
INVALID_PENALTY = 1e3

###############################################################################
# OUTPUT FILES
###############################################################################
STAMP = datetime.now().strftime("%b%d_%H-%M-%S")
RUN_NAME = f"patcharray_{NUM_PATCHES}p_{STAMP}_{socket.gethostname()}"
OUT_DIR = os.path.abspath(RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)
log_file = open(os.path.join(OUT_DIR, RUN_NAME + ".csv"), "w")
log_file.write("eval,num_patches," + ",".join(PARAM_NAMES) +
               ",s11_db,s21_db,weighted_gain,error,status\n")
log_file.flush()

hfss = None
ground_plane = None
eval_count = 0
cache = {}


def mm(*values):
    return [f"{v}mm" for v in values]


def log_row(params, s11, s21, gain, error, status):
    row = [eval_count, NUM_PATCHES, *[f"{p:.4f}" for p in params],
           f"{s11:.3f}", f"{s21:.3f}", f"{gain:.3f}", f"{error:.3f}", status]
    log_file.write(",".join(str(x) for x in row) + "\n")
    log_file.flush()


###############################################################################
# GEOMETRY
###############################################################################
def feed_lengths(params):
    """Return (antenna_length, feed_length, trapezoid_length, rectangle_length)."""
    strip_w, strip_l, patch_w, patch_l, trap_pct = params
    antenna_len = NUM_PATCHES * patch_l + (NUM_PATCHES + 1) * strip_l
    feed_len = 0.5 * (BOARD_LENGTH_MM - antenna_len)
    trap_len = trap_pct * feed_len
    return antenna_len, feed_len, trap_len, feed_len - trap_len


def rect(name, orientation, origin, sizes, color):
    obj = hfss.modeler.create_rectangle(orientation, mm(*origin), mm(*sizes), name=name)
    obj.color = color
    return obj


def build_static_parts():
    """Ground plane, dielectric, radiation boundary, far-field sphere, setup.
    These do not change between iterations."""
    global ground_plane
    gp_w = BOARD_WIDTH_MM + 4
    ground_plane = rect("cell_ground_plane", "XY",
                        (-BOARD_LENGTH_MM / 2, -gp_w / 2, -HEIGHT_MM / 2),
                        (BOARD_LENGTH_MM, gp_w), METAL_COLOR)
    hfss.assign_perfecte_to_sheets([ground_plane.name])

    slab = hfss.modeler.create_box(mm(-BOARD_LENGTH_MM / 2, -BOARD_WIDTH_MM / 2, -HEIGHT_MM / 2),
                                   mm(BOARD_LENGTH_MM, BOARD_WIDTH_MM, HEIGHT_MM),
                                   "dielectric_slab", DIELECTRIC_MATERIAL)
    slab.color = DIELECTRIC_COLOR

    # radiation boundary (positional args work for old and new PyAEDT)
    hfss.create_open_region(FREQ_STR, "Radiation", False, "-z")

    # far-field sphere used for the gain cuts (missing in the original script)
    hfss.insert_infinite_sphere(definition="Theta-Phi",
                                x_start=-180, x_stop=180, x_step=1,   # theta
                                y_start=0, y_stop=90, y_step=90,      # phi = 0 and 90
                                units="deg", name=SPHERE_NAME)

    setup = hfss.create_setup(SETUP_NAME)
    setup.props.update({
        "SolveType": "Single",
        "Frequency": FREQ_STR,
        "MaxDeltaS": 0.1 if SMOKE_TEST else 0.03,
        "MaximumPasses": 3 if SMOKE_TEST else 30,
        "MinimumPasses": 1,
        "MinimumConvergedPasses": 1,
        "PercentRefinement": 30,
        "BasisOrder": 1,
        "DoLambdaRefine": True,
        "DoMaterialLambda": True,
        "SaveAnyFields": True,
    })
    setup.update()
    hfss.modeler.fit_all()


def build_antenna(params):
    """Build strips, patches, feeds and ports. Returns (boundaries, objects) so the
    caller can delete them before the next iteration."""
    strip_w, strip_l, patch_w, patch_l, trap_pct = params
    antenna_len, feed_len, trap_len, rect_len = feed_lengths(params)
    z_top = HEIGHT_MM / 2
    objects, boundaries = [], []

    # --- strip / patch chain ------------------------------------------------
    x = -antenna_len / 2
    for i in range(NUM_PATCHES):
        objects.append(rect(f"transmission_line_{i}", "XY", (x, -strip_w / 2, z_top),
                            (strip_l, strip_w), METAL_COLOR))
        x += strip_l
        objects.append(rect(f"patch_{i}", "XY", (x, -patch_w / 2, z_top),
                            (patch_l, patch_w), METAL_COLOR))
        x += patch_l
    objects.append(rect(f"transmission_line_{NUM_PATCHES}", "XY", (x, -strip_w / 2, z_top),
                        (strip_l, strip_w), METAL_COLOR))   # name was duplicated before

    # --- feeds: board edge -> taper -> rectangle -> antenna -------------------
    half_l = BOARD_LENGTH_MM / 2
    for side, sign in (("0", -1), ("1", +1)):
        edge_x = sign * half_l
        inner_x = edge_x - sign * trap_len
        taper_pts = [[edge_x, -PORT_WIDTH_MM / 2, z_top],
                     [edge_x, PORT_WIDTH_MM / 2, z_top],
                     [inner_x, strip_w / 2, z_top],
                     [inner_x, -strip_w / 2, z_top]]
        taper = hfss.modeler.create_polyline(taper_pts, cover_surface=True, close_surface=True,
                                             name=f"feed_trapezoid_portion_{side}")
        taper.color = METAL_COLOR
        objects.append(taper)

        rect_x0 = inner_x if sign < 0 else antenna_len / 2
        objects.append(rect(f"feed_rectangular_portion_{side}", "XY",
                            (rect_x0, -strip_w / 2, z_top), (rect_len, strip_w), METAL_COLOR))

    boundaries.append(hfss.assign_perfecte_to_sheets([o.name for o in objects]))

    # --- lumped ports at both board edges (ground plane -> trace) -------------
    for idx, sign in ((1, -1), (2, +1)):
        sheet = rect(f"port_{idx}", "YZ", (sign * half_l, -PORT_WIDTH_MM / 2, -HEIGHT_MM / 2),
                     (PORT_WIDTH_MM, HEIGHT_MM), PORT_COLOR)
        objects.append(sheet)
        port = hfss.lumped_port(sheet, reference=ground_plane, create_port_sheet=False,
                                port_on_plane=True, integration_line=0, impedance=50,
                                name=f"port_{idx}_excitation", renormalize=True,
                                deembed=False, terminals_rename=True)
        boundaries.append(port)
    return boundaries, objects


def delete_antenna(boundaries, objects):
    for item in list(boundaries) + list(objects):   # boundaries first, then geometry
        try:
            item.delete()
        except Exception as exc:
            print(f"  (cleanup) could not delete {getattr(item, 'name', item)}: {exc}")


###############################################################################
# RESULTS
###############################################################################
def terminal_names():
    for attr in ("excitation_names", "excitations"):
        try:
            names = list(getattr(hfss, attr))
            if names:
                return names
        except Exception:
            pass
    raise RuntimeError("Could not read excitation/terminal names from HFSS")


def values_of(solution, expr):
    try:  # newer PyAEDT
        _, vals = solution.get_expression_data(expr, formula="real")
        return np.asarray(vals, dtype=float).ravel()
    except AttributeError:  # older PyAEDT
        return np.asarray(solution.data_real(expr), dtype=float).ravel()


def read_results():
    names = terminal_names()
    t1 = next(n for n in names if "port_1" in n)
    t2 = next(n for n in names if "port_2" in n)
    sweep = f"{SETUP_NAME} : LastAdaptive"

    s11_expr, s21_expr = f"dB(St({t1},{t1}))", f"dB(St({t2},{t1}))"
    s11 = values_of(hfss.post.get_solution_data(expressions=s11_expr, setup_sweep_name=sweep,
                                                report_category="Terminal S Parameter"), s11_expr)
    s21 = values_of(hfss.post.get_solution_data(expressions=s21_expr, setup_sweep_name=sweep,
                                                report_category="Terminal S Parameter"), s21_expr)

    gain_expr = "dB(GainTheta)"
    cuts = {}
    for phi in (0, 90):
        sol = hfss.post.get_solution_data(expressions=gain_expr, setup_sweep_name=sweep,
                                          variations={"Freq": FREQ_STR, "Phi": f"{phi}deg"},
                                          primary_sweep_variable="Theta",
                                          report_category="Far Fields", context=SPHERE_NAME)
        cuts[phi] = (np.asarray(sol.primary_sweep_values, dtype=float), values_of(sol, gain_expr))
    return float(np.mean(s11)), float(np.mean(s21)), cuts


def objective_from_results(s11_db, s21_db, cuts):
    theta, gain0 = cuts[0]
    keep = (theta > -180) & (theta < 180)
    theta, gain0 = theta[keep], gain0[keep]
    # weight centered on broadside (theta = 0), peak weight 20, width ~10 samples
    w = windows.general_gaussian(theta.size, p=0.5, sig=10, sym=True)
    w = 20 * (w - w.min()) / w.max()
    weighted_gain = float(np.mean(w * gain0))
    # lower is better: high broadside gain, low S11, low S21 (power radiated, not passed through)
    error = -20 * weighted_gain + 3 * s11_db + s21_db
    return weighted_gain, error, (theta, w)


def save_plots(cuts, window, s11_db, s21_db):
    theta_w, w = window
    fig, ax = plt.subplots()
    for phi, (theta, gain) in cuts.items():
        ax.plot(theta, gain, label=f"Phi={phi} deg")
    ax.plot(theta_w, w, "--", label="weight window")
    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title(f"{FREQ_GHZ:g} GHz, eval {eval_count}: S11={s11_db:.1f} dB, S21={s21_db:.1f} dB")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, f"gain_eval{eval_count:03d}.png"), dpi=120)
    plt.close(fig)

    try:  # 3D screenshot needs pyvista; skip quietly if it is not installed
        plot = hfss.plot(show=False, view="xy", plot_air_objects=False,
                         show_legend=False, force_opacity_value=True)
        plot.plot(os.path.join(OUT_DIR, f"model_eval{eval_count:03d}.jpg"))
    except Exception as exc:
        print(f"  (model screenshot skipped: {exc})")


###############################################################################
# ONE EVALUATION = build -> validate -> solve -> read -> delete
###############################################################################
def evaluate(params):
    global eval_count
    params = np.clip(np.asarray(params, dtype=float), [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    key = tuple(np.round(params, 4))
    if key in cache:
        return cache[key]

    eval_count += 1
    print(f"\n=== Eval {eval_count}: " +
          ", ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, params)))

    _, feed_len, _, _ = feed_lengths(params)
    if feed_len < MIN_FEED_MM:
        print(f"  invalid geometry: feed length {feed_len:.3f} mm (antenna longer than board)")
        log_row(params, np.nan, np.nan, np.nan, INVALID_PENALTY, "invalid_geometry")
        cache[key] = INVALID_PENALTY
        return INVALID_PENALTY

    boundaries, objects = build_antenna(params)
    try:
        if not hfss.validate_full_design():
            print("  validation reported problems - check the AEDT message window")
        if MODE == "geometry":
            print("  geometry mode: built and validated, not solving.")
            log_row(params, np.nan, np.nan, np.nan, np.nan, "geometry_only")
            return np.nan

        hfss.save_project()
        hfss.analyze_setup(SETUP_NAME, NUM_CORES, 1, NUM_GPUS)
        s11_db, s21_db, cuts = read_results()
        weighted_gain, error, window = objective_from_results(s11_db, s21_db, cuts)
        print(f"  S11={s11_db:.2f} dB  S21={s21_db:.2f} dB  weighted gain={weighted_gain:.2f}  "
              f"error={error:.2f}")
        log_row(params, s11_db, s21_db, weighted_gain, error, "ok")
        save_plots(cuts, window, s11_db, s21_db)
        cache[key] = error
        return error
    except Exception as exc:
        print(f"  evaluation failed: {exc}")
        log_row(params, np.nan, np.nan, np.nan, INVALID_PENALTY, f"failed: {exc}".replace(",", ";"))
        cache[key] = INVALID_PENALTY
        return INVALID_PENALTY
    finally:
        # in geometry mode keep the model so you can look at it
        if MODE != "geometry":
            delete_antenna(boundaries, objects)


###############################################################################
# MAIN
###############################################################################
def main():
    global hfss
    t0 = datetime.now()
    hfss = Hfss(f"PatchArray_{STAMP}",          # project
                f"PatchArray_HFSS_{STAMP}",     # design
                "Terminal",                     # solution type
                None,                           # setup
                AEDT_VERSION,                   # version
                NON_GRAPHICAL,                  # non-graphical
                True,                           # new desktop session (only ONE is launched)
                False)                          # close_on_exit (handled below)
    try:
        hfss.modeler.model_units = "mm"
        hfss.autosave_disable()
        build_static_parts()

        if MODE in ("geometry", "single"):
            evaluate(INITIAL)
        elif MODE == "optimize":
            result = minimize(evaluate, INITIAL, method="Nelder-Mead", bounds=BOUNDS,
                              options={"maxfev": MAX_EVALS, "disp": True,
                                       "xatol": 1e-3, "fatol": 0.1})
            print("\nBest parameters:")
            for name, val in zip(PARAM_NAMES, result.x):
                print(f"  {name} = {val:.4f}")
            print(f"Best error: {result.fun:.3f}")
        else:
            raise ValueError(f"Unknown MODE {MODE!r}")
    finally:
        log_file.close()
        print(f"\nResults in: {OUT_DIR}\nElapsed: {datetime.now() - t0}")
        if KEEP_AEDT_OPEN:
            hfss.release_desktop(close_projects=False, close_desktop=False)
        else:
            hfss.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    main()

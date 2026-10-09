"""
4-arm sinuous ultra-wideband antenna - HFSS simulation via PyAEDT.

HOW TO USE (details in README.md)
  1. Change the settings in the "RUN SETTINGS" block below if needed.
  2. Run with the Ansys Python (PyCharm interpreter = the bundled runpython).
  3. Start with MODE = "geometry" (seconds), then "smoke" (coarse, fast), then "full".

Every run creates its own folder results/<date>_<mode>/ with:
  geometry_preview.png    picture of the antenna (made before HFSS even starts)
  sinuous.aedt            the HFSS project (open it in AEDT to look around / animate fields)
  sparams.s4p             raw 4-port S-parameters (Touchstone)
  patterns.csv            far-field data (all cuts, all frequencies)
  01..09_*.png            plots: return loss, isolation, VSWR, impedance, Smith, patterns, gain, AR
  summary.md              key numbers + all plots in one page -> weekly report
  log.txt                 everything printed during the run

WHAT IS MODELLED (stage 1 of the plan)
  Sinuous arms on a Taconic TLY disc (same substrate as the group's reference model), fed at the centre by four small lumped ports
  (one per arm). No balun and no cavity yet: post-processing combines the ports in pairs
  (P1&P3 = polarization X, P2&P4 = polarization Y), which is what an ideal balun does.
"""

import csv
import inspect
import math
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sinuous_geometry import SinuousParams, arm_polygons, port_quads, pad_polygon, save_preview  # noqa: E402

###############################################################################
# RUN SETTINGS  <-- change these
###############################################################################
MODE = "geometry"          # "geometry": build + check the model only, no solve (about 1 minute)
                           # "smoke"   : coarse solve to test the whole pipeline (fastest real results)
                           # "full"    : accurate solve for the report (slow, run it overnight if needed)
AEDT_VERSION = "2025.2"    # the VM has Ansys v252
AEDT_INSTALL_DIR = ""      # leave empty to auto-detect; else the folder containing 'ansysedt'
NON_GRAPHICAL = False      # False = HFSS window opens on the remote desktop so you can watch
KEEP_AEDT_OPEN = True      # leave HFSS open at the end so you can inspect / animate
NUM_CORES = 4              # 4 is safe with a basic license; raise it if the VM has HPC licenses

# Antenna
F_LOW_GHZ = 2.0            # group target: about 2-10 GHz (antenna ~70 mm wide; 0.8 GHz needs ~175 mm)
F_HIGH_GHZ = 10.0          # highest frequency
ALPHA_DEG = 45.0           # arm swing angle
DELTA_DEG = 22.5           # arm half-width; 22.5 = self-complementary for 4 arms
TAU = 0.8                  # cell growth ratio

# Substrate: Taconic TLY, as in the group's reference model (Sinuous DF Antenna Nov18).
# Later target is a thin Kapton film: er 3.4, tand 0.002, 0.05-0.125 mm - change these 4 lines.
SUBSTRATE_LABEL = "Taconic TLY"
SUBSTRATE_NAME = "TaconicTLY_sim"
SUBSTRATE_ER = 2.2
SUBSTRATE_TAND = 0.0009
SUBSTRATE_H_MM = 1.575
SUBSTRATE_MARGIN_MM = 5.0  # board extends this far beyond the arms

# Feed: each port is half of a differential pair -> 2 x 100 = 200 ohm per pair, the value used in the
# group's reference model. The Oct 2 smoke run gave its best quadrature-feed S11 at this value.
PORT_IMPEDANCE_OHM = 100.0

SOLVER = {
    "smoke": dict(adapt_ghz=6.0, max_passes=6, delta_s=0.05, sweep_points=101,
                  pattern_freqs=[2.0, 4.0, 6.0, 8.0, 10.0]),
    # full: mesh refined at THREE frequencies (low, mid, high) so the whole 2-10 GHz band is accurate
    "full": dict(adapt_ghz=[3.0, 6.0, 9.0], max_passes=10, delta_s=0.03, sweep_points=201,
                 pattern_freqs=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
}
# Surface-current pictures are made at the mesh-adaptation frequencies (the only solutions HFSS
# always keeps full fields for). Smoke: 3 GHz. Full: 1.5, 4 and 7 GHz -> the active ring moving inward.
SETUP_NAME = "Setup1"
SPHERE_NAME = "FarField"
###############################################################################

PORTS = ["P1", "P2", "P3", "P4"]
_log_file = None


def log(msg=""):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


def call(func, **kwargs):
    """Call a PyAEDT function with argument names that changed between versions.
    Keys may list alternatives, e.g. {"version|specified_version": "2025.2"}: the first name the
    installed PyAEDT accepts is used. Unknown optional names are skipped with a warning."""
    try:
        params = inspect.signature(func).parameters
        accepts_any = any(p.kind == p.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        params, accepts_any = None, True
    resolved = {}
    for key, value in kwargs.items():
        names = key.split("|")
        chosen = names[0] if params is None else next((n for n in names if n in params), None)
        if chosen is None:
            if accepts_any:
                chosen = names[0]
            else:
                log(f"  note: {getattr(func, '__name__', func)} has no argument {names}; skipped")
                continue
        resolved[chosen] = value
    return func(**resolved)


def find_aedt_install():
    """PyAEDT finds AEDT through environment variables (ANSYSEM_ROOT252 / AWP_ROOT252). On the lab
    VM those are not set in a plain terminal, so look in the usual Linux/Windows install folders."""
    vid = AEDT_VERSION[2:4] + AEDT_VERSION[-1]          # "2025.2" -> "252"
    if os.environ.get(f"ANSYSEM_ROOT{vid}") or os.environ.get(f"AWP_ROOT{vid}"):
        return
    candidates = [
        AEDT_INSTALL_DIR,
        f"/opt/Ansys/v{vid}/AnsysEM", f"/opt/ansys/v{vid}/AnsysEM", f"/opt/AnsysEM/v{vid}/Linux64",
        f"/ansys_inc/v{vid}/AnsysEM", f"/usr/ansys_inc/v{vid}/AnsysEM",
        rf"C:\Program Files\ANSYS Inc\v{vid}\AnsysEM", rf"C:\Program Files\AnsysEM\v{vid}\Win64",
    ]
    for c in candidates:
        if not c:
            continue
        for d in (c, os.path.join(c, "AnsysEM")):
            if any(os.path.exists(os.path.join(d, exe)) for exe in ("ansysedt", "ansysedt.exe")):
                os.environ[f"ANSYSEM_ROOT{vid}"] = d
                log(f"Found AEDT {AEDT_VERSION} in {d}")
                return
    log(f"WARNING: AEDT {AEDT_VERSION} install folder not found automatically. "
        f"Set AEDT_INSTALL_DIR at the top of run_sinuous.py to the folder that contains 'ansysedt'.")


def import_hfss():
    find_aedt_install()
    try:
        from ansys.aedt.core import Hfss
    except ImportError:
        from pyaedt import Hfss
    return Hfss


def as_points(xy, z):
    return [[float(x), float(y), float(z)] for x, y in xy]


###############################################################################
# MODEL
###############################################################################
def build_model(hfss, p):
    z_top = SUBSTRATE_H_MM

    log("Creating substrate material and disc ...")
    mat = call(hfss.materials.add_material, **{"name|materialname": SUBSTRATE_NAME})
    mat.permittivity = SUBSTRATE_ER
    mat.dielectric_loss_tangent = SUBSTRATE_TAND
    call(hfss.modeler.create_cylinder, **{
        "orientation|cs_axis": "Z", "origin|position": [0, 0, 0],
        "radius": p.r_out_mm + SUBSTRATE_MARGIN_MM, "height": SUBSTRATE_H_MM,
        "num_sides|numSides": 0, "name": "substrate", "material|matname": SUBSTRATE_NAME})

    log("Drawing the 4 sinuous arms ...")
    metal = []
    for k, poly in enumerate(arm_polygons(p)):
        obj = call(hfss.modeler.create_polyline, **{
            "points|position_list": as_points(poly, z_top),
            "cover_surface": True, "close_surface": True, "name": f"arm_{k + 1}"})
        metal.append(obj.name)

    pad = call(hfss.modeler.create_polyline, **{
        "points|position_list": as_points(pad_polygon(p), z_top),
        "cover_surface": True, "close_surface": True, "name": "feed_pad"})
    metal.append(pad.name)
    call(hfss.assign_perfecte_to_sheets, **{"assignment|sheet_list": metal, "name|sourcename": "arms_PEC"})

    log("Creating the 4 lumped ports ...")
    for k, q in enumerate(port_quads(p)):
        sheet = call(hfss.modeler.create_polyline, **{
            "points|position_list": as_points(q["points"], z_top),
            "cover_surface": True, "close_surface": True, "name": f"port_sheet_{k + 1}"})
        line = [[float(q["line_start"][0]), float(q["line_start"][1]), z_top],
                [float(q["line_stop"][0]), float(q["line_stop"][1]), z_top]]
        port = call(hfss.lumped_port, **{
            "assignment|signal": sheet.name, "reference": None, "create_port_sheet": False,
            "port_on_plane": True, "integration_line": line, "impedance": PORT_IMPEDANCE_OHM,
            "name|portname": PORTS[k], "renormalize": True, "deembed": False})
        if not port:
            raise RuntimeError(f"Port {PORTS[k]} could not be created")

    log(f"Radiation boundary sized for {F_LOW_GHZ:g} GHz (quarter wavelength of air around the antenna) ...")
    call(hfss.create_open_region, **{
        "frequency|Frequency": f"{F_LOW_GHZ:g}GHz", "boundary|Boundary": "Radiation",
        "apply_infinite_ground|ApplyInfiniteGP": False, "gp_axis|GPAXis": "-z"})

    call(hfss.insert_infinite_sphere, **{
        "definition": "Theta-Phi",
        "theta_start|x_start": -180, "theta_stop|x_stop": 180, "theta_step|x_step": 2,
        "phi_start|y_start": 0, "phi_stop|y_stop": 90, "phi_step|y_step": 90,
        "units": "deg", "name": SPHERE_NAME})
    try:
        hfss.modeler.fit_all()
    except Exception:
        pass
    return metal


def adapt_list(s):
    a = s["adapt_ghz"] or 0.75 * F_HIGH_GHZ
    return list(a) if isinstance(a, (list, tuple)) else [a]


def create_setup(hfss, s):
    freqs_adapt = adapt_list(s)
    setup = call(hfss.create_setup, **{"name|setupname": SETUP_NAME})
    setup.props["MaximumPasses"] = s["max_passes"]
    setup.props["MaxDeltaS"] = s["delta_s"]
    setup.props["MinimumConvergedPasses"] = 1 if MODE == "smoke" else 2
    setup.props["PercentRefinement"] = 30
    setup.props["Frequency"] = f"{freqs_adapt[-1]:g}GHz"
    if len(freqs_adapt) > 1:
        setup.props["SolveType"] = "MultiFrequency"
        setup.props["MultipleAdaptiveFreqsSetup"] = {f"{f:g}GHz": [s["delta_s"]] for f in freqs_adapt}
    ok = setup.update()
    if len(freqs_adapt) > 1 and ok is False:
        log("Multi-frequency mesh setup was not accepted; falling back to a single adaptive frequency")
        setup.props["SolveType"] = "Single"
        setup.update()
        freqs_adapt = freqs_adapt[-1:]
    log(f"Solver setup: adapt mesh at {', '.join(f'{f:g}' for f in freqs_adapt)} GHz, "
        f"max {s['max_passes']} passes, max delta S {s['delta_s']}")
    adapt = freqs_adapt

    f1, f2 = round(F_LOW_GHZ * 0.75, 3), round(F_HIGH_GHZ * 1.1, 3)
    log(f"Interpolating sweep {f1:g}-{f2:g} GHz, {s['sweep_points']} points (S-parameters)")
    call(setup.create_frequency_sweep, **{
        "unit": "GHz", "start_frequency|freqstart": f1, "stop_frequency|freqstop": f2,
        "num_of_freq_points": s["sweep_points"], "name|sweepname": "Sweep",
        "save_fields": False, "sweep_type": "Interpolating"})

    freqs = sorted({f for f in s["pattern_freqs"] if f <= F_HIGH_GHZ + 1e-9})
    log(f"Discrete points with saved far fields (patterns): {freqs} GHz")
    call(setup.create_single_point_sweep, **{
        "unit": "GHz", "freq": list(freqs), "name|sweepname": "Patterns",
        "save_single_field": True, "save_fields": True, "save_rad_fields": True})
    return adapt, freqs


###############################################################################
# RESULTS
###############################################################################
def _xy(sol, expr):
    """(primary sweep values, real values) for an expression, for old and new PyAEDT."""
    if hasattr(sol, "get_expression_data"):
        x, y = sol.get_expression_data(expr, formula="real")
        return np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    y = np.asarray(sol.data_real(expr), dtype=float).ravel()
    x = np.asarray(sol.primary_sweep_values, dtype=float).ravel()
    return x, y


def export_sparams(hfss, run_dir):
    from rf_post import read_touchstone
    out = os.path.join(run_dir, "sparams.s4p")
    try:
        res = call(hfss.export_touchstone, **{"setup|setup_name": SETUP_NAME, "sweep|sweep_name": "Sweep",
                                              "output_file|file_name": out})
        path = res if isinstance(res, str) and os.path.exists(res) else out
        freq, s, z0, names = read_touchstone(path)
        log(f"S-parameters exported ({freq.size} points, ports {names}, reference {z0:g} ohm)")
    except Exception as exc:
        log(f"Touchstone export failed ({exc}); reading S-parameters directly instead ...")
        exprs = [f"{part}(S({a},{b}))" for a in PORTS for b in PORTS for part in ("re", "im")]
        sol = hfss.post.get_solution_data(expressions=exprs, setup_sweep_name=f"{SETUP_NAME} : Sweep",
                                          report_category="Modal Solution Data")
        s = np.zeros((0, 4, 4), complex)
        cols = {}
        for e in exprs:
            freq_raw, cols[e] = _xy(sol, e)
        s = np.zeros((freq_raw.size, 4, 4), complex)
        for i, a in enumerate(PORTS):
            for j, b in enumerate(PORTS):
                s[:, i, j] = cols[f"re(S({a},{b}))"] + 1j * cols[f"im(S({a},{b}))"]
        freq = freq_raw * (1e9 if freq_raw.max() < 1e3 else 1.0)
        z0, names = PORT_IMPEDANCE_OHM, PORTS
    np.savez(os.path.join(run_dir, "sparams_raw.npz"), freq_hz=freq, s=s, z0=z0, port_names=np.array(names))


def export_patterns(hfss, run_dir, freqs):
    modes = {
        "X": {"P1:1": ("1W", "0deg"), "P2:1": ("0W", "0deg"), "P3:1": ("1W", "180deg"), "P4:1": ("0W", "0deg")},
        "CP": {"P1:1": ("1W", "0deg"), "P2:1": ("1W", "90deg"), "P3:1": ("1W", "180deg"), "P4:1": ("1W", "270deg")},
    }
    quantities = {"X": [("RealizedGainTotal", "dB(RealizedGainTotal)"), ("RealizedGainTheta", "dB(RealizedGainTheta)"),
                        ("RealizedGainPhi", "dB(RealizedGainPhi)")],
                  "CP": [("RealizedGainRHCP", "dB(RealizedGainRHCP)"), ("RealizedGainLHCP", "dB(RealizedGainLHCP)"),
                         ("AxialRatio", "dB(AxialRatioValue)")]}
    path = os.path.join(run_dir, "patterns.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "freq_ghz", "phi_deg", "theta_deg", "quantity", "value_db"])
        for mode, src in modes.items():
            call(hfss.edit_sources, **{"assignment|sources": src})
            log(f"Far field, mode {mode} ...")
            for f in freqs:
                for phi in (0, 90):
                    try:
                        sol = hfss.post.get_solution_data(
                            expressions=[e for _, e in quantities[mode]],
                            setup_sweep_name=f"{SETUP_NAME} : Patterns",
                            variations={"Freq": [f"{f:g}GHz"], "Phi": [f"{phi}deg"], "Theta": ["All"]},
                            primary_sweep_variable="Theta", report_category="Far Fields", context=SPHERE_NAME)
                        for qname, expr in quantities[mode]:
                            th, val = _xy(sol, expr)
                            if th.size and np.nanmax(np.abs(th)) <= 2 * math.pi + 1e-6:
                                th = np.degrees(th)   # some versions return radians
                            for t, v in zip(th, val):
                                w.writerow([mode, f"{f:g}", phi, f"{t:.2f}", qname, f"{v:.4f}"])
                    except Exception as exc:
                        log(f"  could not read far field at {f:g} GHz, phi={phi}: {exc}")
            fh.flush()
    # leave polarization X active so field plots in AEDT show the X mode
    call(hfss.edit_sources, **{"assignment|sources": modes["X"]})
    log(f"Far-field data saved: {path}")


def export_field_plots(hfss, run_dir, metal, adapt_freqs):
    for f in adapt_freqs:
        try:
            fp = call(hfss.post.create_fieldplot_surface, **{
                "assignment|objlist": metal, "quantity|quantityName": "Mag_Jsurf",
                "setup|setup_name": f"{SETUP_NAME} : LastAdaptive",
                "intrinsics|IntrinsincDict": {"Freq": f"{f:g}GHz", "Phase": "0deg"},
                "plot_name": f"Jsurf_{f:g}GHz"})
            if fp and hasattr(fp, "export_image"):
                fp.export_image(os.path.join(run_dir, f"field_Jsurf_{f:g}GHz.jpg"))
            log(f"Surface-current plot created at {f:g} GHz")
        except Exception as exc:
            log(f"  field plot at {f:g} GHz skipped: {exc}")


def write_run_info(run_dir, p, extra):
    info = {"run_name": os.path.basename(run_dir), "mode": MODE, "f_low_ghz": F_LOW_GHZ, "f_high_ghz": F_HIGH_GHZ,
            "outer_diameter_mm": f"{2 * p.r_out_mm:.1f}", "inner_radius_mm": f"{p.r_in_mm:.2f}",
            "n_cells": f"{p.n_cells:.1f}", "alpha_deg": ALPHA_DEG, "delta_deg": DELTA_DEG, "tau": TAU,
            "substrate": f"{SUBSTRATE_LABEL} (er={SUBSTRATE_ER}, tand={SUBSTRATE_TAND})",
            "substrate_thickness_mm": SUBSTRATE_H_MM, "port_impedance_ohm": PORT_IMPEDANCE_OHM}
    info.update(extra)
    with open(os.path.join(run_dir, "run_info.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value"])
        for k, v in info.items():
            w.writerow([k, v])


###############################################################################
# MAIN
###############################################################################
def main(hfss_factory=None):
    global _log_file
    if MODE not in ("geometry", "smoke", "full"):
        raise ValueError(f"MODE must be 'geometry', 'smoke' or 'full', not {MODE!r}")
    p = SinuousParams(f_low_ghz=F_LOW_GHZ, f_high_ghz=F_HIGH_GHZ, alpha_deg=ALPHA_DEG,
                      delta_deg=DELTA_DEG, tau=TAU)
    run_dir = os.path.join(HERE, "results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + MODE)
    os.makedirs(run_dir, exist_ok=True)
    _log_file = open(os.path.join(run_dir, "log.txt"), "w")
    log(f"Run folder: {run_dir}")
    log(p.summary())
    write_run_info(run_dir, p, {})
    try:
        save_preview(p, os.path.join(run_dir, "geometry_preview.png"), f"Substrate {SUBSTRATE_LABEL} {SUBSTRATE_H_MM} mm")
        log("Geometry preview saved (geometry_preview.png)")
    except Exception as exc:
        log(f"Preview skipped: {exc}")

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    if hfss_factory is None:
        Hfss = import_hfss()
        hfss_factory = lambda: call(Hfss, **{  # noqa: E731
            "project|projectname": f"Sinuous_{stamp}", "design|designname": "SinuousUWB",
            "solution_type": "Modal", "version|specified_version": AEDT_VERSION,
            "non_graphical": NON_GRAPHICAL, "new_desktop|new_desktop_session": True, "close_on_exit": False})
    log(f"Starting HFSS {AEDT_VERSION} ...")
    hfss = hfss_factory()
    project_file = os.path.join(run_dir, "sinuous.aedt")
    try:
        hfss.modeler.model_units = "mm"
        try:
            hfss.autosave_disable()
        except Exception:
            pass
        metal = build_model(hfss, p)
        ok = hfss.validate_full_design()
        log(f"Design validation result: {ok}")
        call(hfss.save_project, **{"file_name|project_file": project_file})

        if MODE == "geometry":
            log("MODE = 'geometry': model built and saved, nothing solved. Look at it in the HFSS window,")
            log("then set MODE = 'smoke' to run the first simulation.")
            return run_dir

        s = SOLVER[MODE]
        adapt, freqs = create_setup(hfss, s)
        call(hfss.save_project, **{"file_name|project_file": project_file})
        log(f"Solving on {NUM_CORES} cores. This is the slow part - watch progress in the HFSS window ...")
        t0 = time.time()
        call(hfss.analyze_setup, **{"name": SETUP_NAME, "cores|num_cores": NUM_CORES, "tasks|num_tasks": 1})
        solve_time = time.time() - t0
        log(f"Solve finished in {solve_time / 60:.1f} min")
        write_run_info(run_dir, p, {"adapt_freq_ghz": " ".join(f"{f:g}" for f in adapt), "max_passes": s["max_passes"],
                                    "solve_time": f"{solve_time / 60:.1f} min"})
        call(hfss.save_project, **{"file_name|project_file": project_file})

        for name, fn in (("S-parameters", lambda: export_sparams(hfss, run_dir)),
                         ("far-field patterns", lambda: export_patterns(hfss, run_dir, freqs)),
                         ("field plots", lambda: export_field_plots(hfss, run_dir, metal, adapt))):
            try:
                fn()
            except Exception:
                log(f"Exporting {name} failed (other results are still saved):\n{traceback.format_exc()}")
        call(hfss.save_project, **{"file_name|project_file": project_file})

        try:
            from rf_post import process_run
            summary, plots = process_run(run_dir)
            log(f"Made {len(plots)} plots. Open {summary} for the report page.")
        except Exception:
            log(f"Plotting failed; the raw data is saved, rerun later with: python rf_post.py \"{run_dir}\"\n"
                f"{traceback.format_exc()}")
        return run_dir
    except Exception:
        log("RUN STOPPED WITH AN ERROR:\n" + traceback.format_exc())
        log("Copy the lines above (and log.txt) into the chat to get it fixed.")
        raise
    finally:
        try:
            if KEEP_AEDT_OPEN:
                hfss.release_desktop(close_projects=False, close_desktop=False)
            else:
                hfss.release_desktop(close_projects=True, close_desktop=True)
        except Exception:
            pass
        log("Done.")
        _log_file.close()
        _log_file = None


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cardiovascular 0D Model Optimization — PAH Severity Spectrum
==============================================================
Fits Regazzoni2020 model parameters to match pre-capillary pulmonary
arterial hypertension across four severity levels:

    mild              WHO FC I / early II     mPAP ~22-25
    moderate          WHO FC II / early III   mPAP ~30-33
    moderate_severe   WHO FC II-III           mPAP ~40-42
    severe            WHO FC III-IV           mPAP ~45+

Run modes
---------
  python optimize_pah.py mild                    # optimise mild PAH
  python optimize_pah.py severe --n-trials 5000  # severe with more trials
  python optimize_pah.py moderate --eval-only    # evaluate saved JSON only
  python optimize_pah.py all                     # run all four severities

Outputs (per severity)
----------------------
  optimized_regazzoni_pah_{severity}.json     best parameters
  pv_evaluation_pah_{severity}.png            evaluation figure
  pah_{severity}.db                           Optuna study (SQLite)
"""

import argparse
import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
from matplotlib.gridspec import GridSpec
from optuna.samplers import TPESampler

from circulation.regazzoni2020 import Regazzoni2020

# suppress noisy loggers
logging.getLogger("circulation.base").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==============================================================================
# CLINICAL TARGETS BY SEVERITY
# ==============================================================================
SEVERITY_TARGETS = {
    "mild": {
        "RV_ESP":    38.0,    # mPAP ~22-25
        "RV_EDP":    6.0,
        "LA_P_MEAN": 9.0,
        "SV":        65.0,
        "LV_ESP":    118.0,
        "Ao_DBP":    80.0,
        "RV_EDV":    155.0,
        "LV_EDV":    130.0,
    },
    "moderate": {
        "RV_ESP":    50.0,    # mPAP ~30-33
        "RV_EDP":    8.0,
        "LA_P_MEAN": 9.0,
        "SV":        60.0,
        "LV_ESP":    115.0,
        "Ao_DBP":    75.0,
        "RV_EDV":    160.0,
        "LV_EDV":    115.0,
    },
    "moderate_severe": {
        "RV_ESP":    63.0,    # mPAP ~40-42
        "RV_EDP":    10.0,
        "LA_P_MEAN": 8.5,
        "SV":        52.0,
        "LV_ESP":    105.0,
        "Ao_DBP":    72.0,
        "RV_EDV":    180.0,
        "LV_EDV":    105.0,
    },
    "severe": {
        "RV_ESP":    72.0,    # mPAP ~45+
        "RV_EDP":    11.0,
        "LA_P_MEAN": 8.5,
        "SV":        46.0,
        "LV_ESP":    102.0,
        "Ao_DBP":    70.0,
        "RV_EDV":    198.0,
        "LV_EDV":    100.0,
    },
}

# Pass/fail tolerances (not used in cost function)
def _tolerances(targets):
    tol = {k: 15.0 for k in targets}
    tol["RV_EDP"] = 25.0
    tol["LV_ESP"] = 20.0
    tol["LV_EDV"] = 20.0
    tol["Ao_DBP"] = 20.0
    return tol


# ==============================================================================
# PARAMETER CONFIG — shared across all severities
#   (path, initial_guess, lower_bound, upper_bound, scale, note)
#   real_value = suggested_value * scale
# ==============================================================================
PARAMETER_CONFIG = [
    # -- Right heart ---------------------------------------------------------
    ("chambers.RV.EA",          0.8,    0.3,    2.5,    1.0,   "RV max elastance"),
    ("chambers.RV.EB",          0.06,   0.01,   0.25,   0.1,   "RV passive elastance"),
    ("chambers.RV.V0",          60.0,   20.0,   120.0,  1.0,   "RV unstressed volume [mL]"),
    ("chambers.RV.TC",          0.25,   0.18,   0.35,   1.0,   "RV contraction duration [s]"),
    ("chambers.RV.TR",          0.40,   0.28,   0.45,   1.0,   "RV relaxation duration [s]"),

    # -- Left heart (preserved) ----------------------------------------------
    ("chambers.LV.EA",          2.8,    1.5,    6.0,    1.0,   "LV max elastance"),
    ("chambers.LV.EB",          0.09,   0.04,   0.30,   0.1,   "LV passive elastance"),
    ("chambers.LV.V0",          42.0,   25.0,   55.0,   1.0,   "LV unstressed volume [mL]"),
    ("chambers.LV.TC",          0.25,   0.18,   0.30,   1.0,   "LV contraction duration [s]"),
    ("chambers.LV.TR",          0.40,   0.28,   0.42,   1.0,   "LV relaxation duration [s]"),
    ("chambers.LA.EB",          0.05,   0.01,   0.15,   0.1,   "LA passive elastance"),
    ("chambers.LA.EA",          0.07,   0.02,   0.15,   0.1,   "LA active elastance"),

    # -- Valves --------------------------------------------------------------
    ("valves.MV.Rmin",          0.01,   0.001,  0.08,   0.01,  "Mitral valve min resistance"),

    # -- Systemic circulation ------------------------------------------------
    ("circulation.SYS.R_AR",   0.95,   0.4,    2.0,    1.0,   "Systemic arterial resistance"),
    ("circulation.SYS.C_AR",   1.2,    0.4,    3.5,    1.0,   "Systemic arterial compliance"),
    ("circulation.SYS.C_VEN",  40.0,   10.0,   100.0,  10.0,  "Systemic venous compliance"),

    # -- Pulmonary circulation -----------------------------------------------
    ("circulation.PUL.R_AR",   0.20,   0.05,   0.8,    0.1,   "Pulmonary arterial resistance"),
    ("circulation.PUL.C_AR",   0.50,   0.10,   1.0,    0.1,   "Pulmonary arterial compliance"),
    ("circulation.PUL.R_VEN",  0.15,   0.01,   1.0,    0.01,  "Pulmonary venous resistance"),
    ("circulation.PUL.C_VEN",  40.0,   5.0,    200.0,  10.0,  "Pulmonary venous compliance"),

    # -- Blood volume --------------------------------------------------------
    ("TOTAL_VOLUME_OFFSET",     300.0,  0.0,    2000.0, 100.0, "Total volume offset [mL]"),
]


# ==============================================================================
# HELPERS
# ==============================================================================
def find_true_edp(v, p, tol_ml=1.0):
    """Lowest pressure within tol_ml of peak volume."""
    v_max = np.max(v)
    candidates = np.where(v >= v_max - tol_ml)[0]
    idx = candidates[np.argmin(p[candidates])]
    return idx, p[idx]


def extract_metrics(p_lv, v_lv, p_rv, v_rv, p_ao, p_la):
    _, lv_edp = find_true_edp(v_lv, p_lv)
    _, rv_edp = find_true_edp(v_rv, p_rv)
    return {
        "LV_ESP":    np.max(p_lv),
        "LV_EDP":    lv_edp,
        "SV":        np.max(v_lv) - np.min(v_lv),
        "RV_ESP":    np.max(p_rv),
        "RV_EDP":    rv_edp,
        "Ao_DBP":    np.min(p_ao),
        "LA_P_MEAN": np.mean(p_la),
        "RV_EDV":    np.max(v_rv),
        "LV_EDV":    np.max(v_lv),
    }


def _rel(val, target):
    return abs(val - target) / max(1.0, abs(target))


def compute_cost(metrics, targets):
    T = targets
    c  = 200.0 * _rel(metrics["SV"], T["SV"])
    c += 150.0 * _rel(metrics["RV_EDV"], T["RV_EDV"])
    c += 150.0 * _rel(metrics["LV_EDV"], T["LV_EDV"])
    c += 150.0 * _rel(metrics["RV_ESP"], T["RV_ESP"])
    c += 60.0  * _rel(metrics["RV_EDP"], T["RV_EDP"])
    c += 150.0 * _rel(metrics["LV_ESP"], T["LV_ESP"])
    c += 100.0 * _rel(metrics["Ao_DBP"], T["Ao_DBP"])
    c += 150.0 * _rel(metrics["LA_P_MEAN"], T["LA_P_MEAN"])
    return c


# ==============================================================================
# MODEL INTERFACE
# ==============================================================================
class ModelInterface:
    def __init__(self):
        base = Regazzoni2020(add_units=False)
        self.base_params = base.parameters
        self.base_init   = base._initial_state

    def build(self, suggested: dict):
        params     = self.base_params.copy()
        init_state = self.base_init.copy()

        for key, _, _, _, scale, _ in PARAMETER_CONFIG:
            real_val = suggested[key] * scale

            if key == "TOTAL_VOLUME_OFFSET":
                C_ven = params["circulation"]["SYS"]["C_VEN"]
                p0 = init_state["p_VEN_SYS"]
                if hasattr(p0, "magnitude"):
                    p0 = float(p0.magnitude)
                init_state["p_VEN_SYS"] = p0 + real_val / C_ven
            else:
                parts = key.split(".")
                d = params
                for k in parts[:-1]:
                    d = d[k]
                d[parts[-1]] = real_val

        return params, init_state


# ==============================================================================
# OBJECTIVE FACTORY
# ==============================================================================
def make_objective(interface, targets):
    def objective(trial):
        suggested = {
            key: trial.suggest_float(key, lb / scale, ub / scale)
            for key, _, lb, ub, scale, _ in PARAMETER_CONFIG
        }

        params, init_state = interface.build(suggested)

        lv = params["chambers"]["LV"]
        if lv["TC"] + lv["TR"] + lv.get("tC", 0.1) > 0.78:
            raise optuna.exceptions.TrialPruned()

        model = Regazzoni2020(parameters=params, initial_state=init_state,
                              add_units=False, verbose=False)
        try:
            history = model.solve(num_beats=15, dt=2e-3)
        except Exception:
            raise optuna.exceptions.TrialPruned()

        samples = int((1 / params["HR"]) / 2e-3)
        slc = slice(-samples, None)

        p_lv, v_lv = history["p_LV"][slc], history["V_LV"][slc]
        p_rv, v_rv = history["p_RV"][slc], history["V_RV"][slc]
        p_ao       = history["p_AR_SYS"][slc]
        p_la       = history["p_LA"][slc]

        if np.max(p_rv) > 300.0 or np.isnan(np.sum(p_rv)):
            raise optuna.exceptions.TrialPruned()

        metrics = extract_metrics(p_lv, v_lv, p_rv, v_rv, p_ao, p_la)
        return compute_cost(metrics, targets)

    return objective


# ==============================================================================
# OPTIMISATION
# ==============================================================================
def run_optimization(severity, n_trials, seed=42):
    targets    = SEVERITY_TARGETS[severity]
    study_name = f"pah_{severity}"
    out_json   = f"optimized_regazzoni_pah_{severity}.json"

    print("\n" + "=" * 65)
    print(f"  Optimising: {severity} PAH  ({n_trials} trials, TPE sampler)")
    print("=" * 65)
    print("\nTargets:")
    for k, v in targets.items():
        print(f"  {k:<14}: {v}")

    interface = ModelInterface()

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=False,
        direction="minimize",
        sampler=TPESampler(n_startup_trials=100, seed=seed),
    )

    study.optimize(
        make_objective(interface, targets),
        n_trials=n_trials,
        n_jobs=-1,
        show_progress_bar=True,
    )

    print(f"\n  Best cost: {study.best_value:.4f}")

    # Reconstruct best parameters and verify at high resolution
    params, init_state = interface.build(study.best_params)
    model = Regazzoni2020(parameters=params, initial_state=init_state,
                          add_units=False, verbose=False)
    history = model.solve(num_beats=20, dt=1e-3)

    samples = int((1 / params["HR"]) / 1e-3)
    slc = slice(-samples, None)
    metrics = extract_metrics(
        history["p_LV"][slc], history["V_LV"][slc],
        history["p_RV"][slc], history["V_RV"][slc],
        history["p_AR_SYS"][slc], history["p_LA"][slc],
    )

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    save_data = {
        "description": f"Optimised {severity} PAH",
        "severity": severity,
        "targets": targets,
        "metrics_achieved": metrics,
        "parameters": params,
        "initial_state": init_state,
    }
    with open(out_json, "w") as f:
        json.dump(save_data, f, cls=_NpEncoder, indent=4)
    print(f"\n  Saved: {out_json}")

    return out_json


# ==============================================================================
# EVALUATION
# ==============================================================================
def evaluate_and_plot(severity, json_path):
    targets = SEVERITY_TARGETS[severity]
    tols    = _tolerances(targets)

    print(f"\nEvaluating {json_path} ({severity} PAH) ...")

    with open(json_path) as f:
        data = json.load(f)

    parameters    = data["parameters"]
    initial_state = data["initial_state"]

    model   = Regazzoni2020(parameters=parameters, initial_state=initial_state,
                            add_units=False, verbose=False)
    history = model.solve(num_beats=15, dt=2e-3)

    samples = int((1 / parameters["HR"]) / 2e-3)
    slc     = slice(-samples, None)

    v_lv = history["V_LV"][slc];    p_lv = history["p_LV"][slc]
    v_rv = history["V_RV"][slc];    p_rv = history["p_RV"][slc]
    p_ao = history["p_AR_SYS"][slc]
    p_la = history["p_LA"][slc]
    t    = np.linspace(0, 1 / parameters["HR"], samples)

    m = extract_metrics(p_lv, v_lv, p_rv, v_rv, p_ao, p_la)

    # -- Console table -----------------------------------------------------
    print("\n" + "=" * 68)
    print(f"{'METRIC':<14} | {'TARGET':>8} | {'ACHIEVED':>9} | {'ERROR':>8} | {'TOL':>5} | STATUS")
    print("-" * 68)
    all_pass = True
    for key, tgt in targets.items():
        val = m[key]
        err = (val - tgt) / tgt * 100
        tol = tols[key]
        ok  = abs(err) <= tol
        if not ok:
            all_pass = False
        flag = "PASS" if ok else "FAIL"
        print(f"{key:<14} | {tgt:>8.1f} | {val:>9.2f} | {err:>+7.1f}% | +/-{tol:>3.0f}% | {flag}")
    print("-" * 68)
    print(f"Overall: {'ALL METRICS PASS' if all_pass else 'SOME METRICS FAILED'}")
    print("=" * 68)

    # -- Figure ------------------------------------------------------------
    label = severity.replace("_", "-").title()

    fig = plt.figure(figsize=(17, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)

    ax_rv  = fig.add_subplot(gs[0, 0])
    ax_lv  = fig.add_subplot(gs[0, 1])
    ax_ao  = fig.add_subplot(gs[0, 2])
    ax_tbl = fig.add_subplot(gs[1, :])

    # RV PV loop
    ax_rv.plot(v_rv, p_rv, 'b-', lw=2.5)
    ax_rv.fill(v_rv, p_rv, alpha=0.12, color='blue')
    ax_rv.axhline(targets["RV_ESP"], color='navy',           ls='--', lw=1.2,
                  label=f"Target ESP {targets['RV_ESP']:.0f}")
    ax_rv.axhline(targets["RV_EDP"], color='steelblue',      ls=':', lw=1.2,
                  label=f"Target EDP {targets['RV_EDP']:.0f}")
    ax_rv.axvline(targets["RV_EDV"], color='cornflowerblue', ls='--', lw=1.2,
                  label=f"Target EDV {targets['RV_EDV']:.0f}")
    ax_rv.set_title(f"RV PV Loop ({label} PAH)\n"
                    f"ESP {m['RV_ESP']:.1f}  EDP {m['RV_EDP']:.1f}  EDV {m['RV_EDV']:.1f}",
                    fontsize=9)
    ax_rv.set_xlabel("Volume [mL]"); ax_rv.set_ylabel("Pressure [mmHg]")
    ax_rv.legend(fontsize=7.5); ax_rv.grid(True, alpha=0.3)

    # LV PV loop
    ax_lv.plot(v_lv, p_lv, 'r-', lw=2.5)
    ax_lv.fill(v_lv, p_lv, alpha=0.12, color='red')
    ax_lv.axhline(targets["LV_ESP"], color='darkred', ls='--', lw=1.2,
                  label=f"Target ESP {targets['LV_ESP']:.0f}")
    ax_lv.axvline(targets["LV_EDV"], color='tomato',  ls='--', lw=1.2,
                  label=f"Target EDV {targets['LV_EDV']:.0f}")
    ax_lv.set_title(f"LV PV Loop\n"
                    f"ESP {m['LV_ESP']:.1f}  EDV {m['LV_EDV']:.1f}  SV {m['SV']:.1f}",
                    fontsize=9)
    ax_lv.set_xlabel("Volume [mL]"); ax_lv.set_ylabel("Pressure [mmHg]")
    ax_lv.legend(fontsize=7.5); ax_lv.grid(True, alpha=0.3)

    # Aortic + LA traces
    ax_ao.plot(t, p_ao, 'k-',  lw=2,   label=f"Ao (DBP {m['Ao_DBP']:.1f})")
    ax_ao.plot(t, p_la, 'm--', lw=1.5, label=f"LA (mean {m['LA_P_MEAN']:.1f})")
    ax_ao.axhline(targets["Ao_DBP"],    color='gray',   ls='--', lw=1)
    ax_ao.axhline(targets["LA_P_MEAN"], color='violet', ls=':',  lw=1)
    ax_ao.set_title(f"Aortic & LA Pressure\n"
                    f"Ao_DBP {m['Ao_DBP']:.1f}  LA_mean {m['LA_P_MEAN']:.1f}",
                    fontsize=9)
    ax_ao.set_xlabel("Time [s]"); ax_ao.set_ylabel("Pressure [mmHg]")
    ax_ao.legend(fontsize=7.5); ax_ao.grid(True, alpha=0.3)

    # Summary table
    ax_tbl.axis('off')
    col_labels = ["Metric", "Target", "Achieved", "Error %", "Tolerance", "Status"]
    rows, cell_colors = [], []
    for key, tgt in targets.items():
        val = m[key]
        err = (val - tgt) / tgt * 100
        tol = tols[key]
        ok  = abs(err) <= tol
        rows.append([key, f"{tgt:.1f}", f"{val:.2f}", f"{err:+.2f}%",
                     f"+/-{tol:.0f}%", "PASS" if ok else "FAIL"])
        cell_colors.append(["#f5f5f5"] * 5 + ["#c6efce" if ok else "#ffc7ce"])

    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       cellColours=cell_colors, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4472c4')
            cell.set_text_props(color='white', fontweight='bold')

    ax_tbl.set_title(f"{label} PAH — Targets vs Achieved",
                     fontsize=12, fontweight='bold', pad=8)

    out_fig = f"pv_evaluation_pah_{severity}.png"
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_fig}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Optimise Regazzoni2020 0D model for PAH severity spectrum.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Severities: mild, moderate, moderate_severe, severe, all",
    )
    parser.add_argument(
        "severity",
        choices=["mild", "moderate", "moderate_severe", "severe", "all"],
        help="PAH severity level to optimise (or 'all' for the full spectrum).",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip optimisation and evaluate the saved JSON directly.",
    )
    parser.add_argument(
        "--json", default=None,
        help="JSON file to evaluate (default: auto-generated name).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=3000,
        help="Number of Optuna trials (default: 3000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for TPE sampler (default: 42).",
    )
    args = parser.parse_args()

    severities = list(SEVERITY_TARGETS.keys()) if args.severity == "all" else [args.severity]

    for sev in severities:
        out_json = args.json or f"optimized_regazzoni_pah_{sev}.json"

        if not args.eval_only:
            out_json = run_optimization(sev, args.n_trials, args.seed)

        evaluate_and_plot(sev, out_json)


if __name__ == "__main__":
    main()

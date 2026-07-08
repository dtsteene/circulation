#!/usr/bin/env python3
"""
Mesh-Specific Circulation Optimization — v11
=============================================
Unified optimizer that tunes the 0D Regazzoni 2020 circulation model to a
spectrum of RV systolic pressures, using the FEM mesh's cavity volumes as
fixed geometric constraints.

v11 target-framework rewrite (2026-04-22)
------------------------------------------
The spectrum's ONE independent variable is RV_ESP — this drives the
transmural septal pressure collapse (P_LV_ES - P_RV_ES → 0 at end-stage PAH),
which is the central mechanical finding of the thesis. All other hemodynamic
variables are physiological guardrails: either held constant at healthy-normal
values (LV_ESP, LV_EDP, Ao_DBP, LA_P_MEAN, LVEF_floor), or assigned via three
Humbert 2022 ESC/ERS risk-stratum bands (RV_EDP, CI_min, RVEF_floor).

Every target is traceable to a specific primary source (Humbert 2022,
Kovacs 2009, Benza 2010/2012, Tello 2019) — see:
  /global/D1/homes/dtsteene/cardiac-work/results/docs/target_grounding_audit.md

Changed from v10:
  - LV_ESP, LV_EDP, Ao_DBP, LA_P_MEAN now held constant (v10 per-severity
    declines had no primary source).
  - LV_EF_floor held at 0.55 for all severities (LVEF not prognostic per
    Baggen 2016).
  - RV_EDP, RVEF_floor, CI_min assigned in three ESC bands (Humbert T16).
  - CO_min (L/min) replaced by CI_min (L/min/m²) to match ESC convention.
    Cost function now multiplies CI by --bsa (default 1.75 m²).
  - Optuna study names prefixed "v11_" so they do not collide with v10 DBs.

Run modes
---------
  python optimize_mesh_circ.py --mesh ukb --severity healthy
  python optimize_mesh_circ.py --mesh /path/to/healthy.xdmf --severity moderate
  python optimize_mesh_circ.py --mesh ukb --severity all
  python optimize_mesh_circ.py --mesh ukb --severity healthy --eval-only
  python optimize_mesh_circ.py --mesh ukb --severity all --bsa 1.82

Outputs (per severity, in results_mesh_{mesh}/)
------------------------------------------------
  optimized_regazzoni_{mesh}_{severity}.json   best parameters
  pv_evaluation_{mesh}_{severity}.png          evaluation figure
  v11_mesh_{mesh}_{severity}.db                Optuna study (SQLite)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from matplotlib.gridspec import GridSpec
from optuna.samplers import TPESampler, CmaEsSampler

from circulation.regazzoni2020 import Regazzoni2020

# suppress noisy loggers
logging.getLogger("circulation.base").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==============================================================================
# MESH VOLUME READER
# ==============================================================================
UNIT_TO_ML = {
    "mm": 1.0 / 1000.0,   # mm³ → mL
    "cm": 1.0,             # cm³ = mL
    "m":  1e6,             # m³  → mL
}

# Default body surface area for CI calculation.
# 1.75 m² is a reasonable adult mean; override via --bsa for patient-specific runs.
# Used to convert simulated absolute CO (L/min) into CI (L/min/m²) for
# comparison with Humbert 2022 Table 16 risk-stratum cutoffs.
DEFAULT_BSA_M2 = 1.75


def read_mesh_volumes(mesh_path, mesh_unit=None):
    """Compute LV and RV end-diastolic cavity volumes from a mesh.

    Parameters
    ----------
    mesh_path : str or Path
        Path to an XDMF mesh file (.xdmf), a cardiac_geometries geometry
        folder (containing geometry.bp), or the keyword ``ukb`` to generate
        the synthetic UK Biobank atlas mesh.
    mesh_unit : str or None
        Coordinate unit of the mesh: ``"mm"``, ``"cm"``, or ``"m"``.
        If None, defaults to ``"mm"`` for ``ukb`` and directory paths,
        and ``"cm"`` for XDMF files (legacy behaviour).

    Returns
    -------
    lv_edv_ml, rv_edv_ml : float
        Cavity volumes in mL.
    mesh_name : str
        Identifier derived from the mesh path.
    """
    from mpi4py import MPI
    import dolfinx
    import ufl

    def _volumes(mesh, ffun, lv_tag, rv_tag, to_ml):
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=ffun)
        x = ufl.SpatialCoordinate(mesh)
        n = ufl.FacetNormal(mesh)
        form = (1.0 / 3.0) * ufl.dot(x, n)
        lv = abs(float(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(form * ds(lv_tag)))))
        rv = abs(float(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(form * ds(rv_tag)))))
        return lv * to_ml, rv * to_ml

    p = str(mesh_path)
    comm = MPI.COMM_SELF

    if p.lower() == "ukb":
        unit = mesh_unit or "mm"
        to_ml = UNIT_TO_ML[unit]
        import tempfile
        import cardiac_geometries.mesh
        with tempfile.TemporaryDirectory() as tmp:
            geo = cardiac_geometries.mesh.ukb(
                outdir=Path(tmp), comm=comm, case="ED",
                char_length_max=5.0, char_length_min=5.0, clipped=True,
            )
            lv, rv = _volumes(
                geo.mesh, geo.ffun,
                geo.markers["ENDO_LV"][0], geo.markers["ENDO_RV"][0],
                to_ml,
            )
        return lv, rv, "ukb"

    p_path = Path(p)

    if p_path.suffix == ".xdmf":
        unit = mesh_unit or "cm"
        to_ml = UNIT_TO_ML[unit]
        with dolfinx.io.XDMFFile(comm, p, "r") as xdmf:
            mesh = xdmf.read_mesh(name="mesh")
            mesh.topology.create_connectivity(
                mesh.topology.dim - 1, mesh.topology.dim)
            try:
                ffun = xdmf.read_meshtags(mesh, name="facet_tags")
            except RuntimeError:
                ffun = xdmf.read_meshtags(mesh, name="mesh_tags")
        lv, rv = _volumes(mesh, ffun, 30, 20, to_ml)
        return lv, rv, p_path.stem

    if p_path.is_dir():
        unit = mesh_unit or "mm"
        to_ml = UNIT_TO_ML[unit]
        import cardiac_geometries.geometry
        geo = cardiac_geometries.geometry.Geometry.from_folder(
            comm=comm, folder=p_path)
        # Handle both marker naming conventions
        for lv_name, rv_name in [("ENDO_LV", "ENDO_RV"), ("LV", "RV")]:
            if lv_name in geo.markers:
                lv_tag = geo.markers[lv_name][0]
                rv_tag = geo.markers[rv_name][0]
                break
        else:
            raise KeyError(f"No LV/RV endo markers found. Available: {list(geo.markers)}")
        lv, rv = _volumes(geo.mesh, geo.ffun, lv_tag, rv_tag, to_ml)
        return lv, rv, p_path.name

    raise ValueError(f"Unsupported mesh path: {p}")


# ==============================================================================
# SEVERITY — PRESSURE TARGETS + PHYSIOLOGICAL GUARDRAILS (v11)
# ==============================================================================
# Provenance: every target traces to a specific table in a specific paper.
# See: /global/D1/homes/dtsteene/cardiac-work/results/docs/target_grounding_audit.md
#
# The spectrum's ONE independent variable is RV_ESP — this drives the
# transmural septal pressure collapse (P_LV_ES - P_RV_ES → 0 at end-stage),
# which is the central mechanical finding of the thesis.
#
# Everything else is a physiological guardrail: either held constant at a
# healthy value (because pre-capillary PAH does not alter that variable) or
# assigned one of three ESC risk-stratum bands (low / intermediate / high).
#
# RV_ESP anchor points:
#   healthy (22):     Kovacs 2009 Table 1, p.890 — sPAP mean 20.8 ± 4.4 (n=625)
#   borderline (30):  Kovacs 2009 p.890 text — sPAP ULN (mean + 2SD) = 29.6
#   severe (75):      Tello 2019 Table 2, p.4 — PASP 75 ± 24 (severe-PH cohort)
#   (mild 45, moderate 55, moderate_severe 65, very_severe 85, end_stage 95:
#    interpolated/extrapolated between the three anchors)
#
# Held-constant targets (no per-severity variation):
#   LV_ESP = 120:    Benza 2010 Table 1, p.166 — SBP 116 ± 17 mmHg (n=2716,
#                    full PAH cohort, mostly WHO-FC II-III). Systemic BP is
#                    preserved in pre-capillary PAH.
#   LV_EDP = 8:      Kovacs 2009 Table 1, p.890 — PAWP 8.0 ± 2.9 (n=882).
#                    LVEDP ≈ PAWP in pre-capillary PAH.
#   Ao_DBP = 80:     Consistent with Benza 2010 preserved SBP; conventional
#                    adult normal DBP. No paper supports a decline.
#   LA_P_MEAN = 8:   Kovacs 2009 Table 1 — PAWP 8.0. Humbert 2022 Table 5
#                    (p.20) bounds pre-capillary PH at PAWP ≤ 15.
#
# Prior v10 ramps LV_ESP (118→90), LV_EDP (8→4), Ao_DBP (80→60),
# LA_P_MEAN (8→7), and LV_EF_floor (0.55→0.20) had no primary source
# in the cited literature and have been dropped. See audit document.

SEVERITY_PRESSURE_TARGETS = {
    # RV_ESP is the independent variable of the study; everything else is constant.
    "healthy_low":     {"RV_ESP": 22.0, "RV_EDP": 5.0,  "LV_ESP": 120.0, "LV_EDP": 8.0, "Ao_DBP": 80.0, "LA_P_MEAN": 8.0},
    "healthy":         {"RV_ESP": 22.0, "RV_EDP": 5.0,  "LV_ESP": 120.0, "LV_EDP": 8.0, "Ao_DBP": 80.0, "LA_P_MEAN": 8.0},
    "borderline":      {"RV_ESP": 30.0, "RV_EDP": 5.0,  "LV_ESP": 120.0, "LV_EDP": 8.0, "Ao_DBP": 80.0, "LA_P_MEAN": 8.0},
    "mild":            {"RV_ESP": 45.0, "RV_EDP": 5.0,  "LV_ESP": 120.0, "LV_EDP": 8.0, "Ao_DBP": 80.0, "LA_P_MEAN": 8.0},
    "moderate":        {"RV_ESP": 55.0, "RV_EDP": 10.0, "LV_ESP": 120.0, "LV_EDP": 8.0, "Ao_DBP": 80.0, "LA_P_MEAN": 8.0},
    "moderate_severe": {"RV_ESP": 65.0, "RV_EDP": 10.0, "LV_ESP": 120.0, "LV_EDP": 8.0, "Ao_DBP": 80.0, "LA_P_MEAN": 8.0},
    "severe":          {"RV_ESP": 75.0, "RV_EDP": 16.0, "LV_ESP": 120.0, "LV_EDP": 8.0, "Ao_DBP": 80.0, "LA_P_MEAN": 8.0},
    "very_severe":     {"RV_ESP": 85.0, "RV_EDP": 16.0, "LV_ESP": 120.0, "LV_EDP": 8.0, "Ao_DBP": 80.0, "LA_P_MEAN": 8.0},
    "end_stage":       {"RV_ESP": 95.0, "RV_EDP": 16.0, "LV_ESP": 120.0, "LV_EDP": 8.0, "Ao_DBP": 80.0, "LA_P_MEAN": 8.0},
}

# Physiological guardrails — three ESC risk bands from Humbert 2022 Table 16, p.45.
# (Three-strata model; 1-year mortality low <5% / intermediate 5-20% / high >20%.)
#
#   Low-risk band:          RAP <8,   CI ≥2.5,  RVEF >54%
#   Intermediate band:      RAP 8-14, CI 2.0-2.4, RVEF 37-54%
#   High-risk band:         RAP >14,  CI <2.0,  RVEF <37%
#
# Severity-to-band assignment:
#   Low:          healthy_low, healthy, borderline, mild
#   Intermediate: moderate, moderate_severe
#   High:         severe, very_severe, end_stage
#
# LVEF floor held at 0.55 throughout: LVEF is not prognostic in PAH
# (Baggen 2016 Eur Radiol 26:3771–3780, pooled HR 1.06 p=0.301). No primary
# source publishes per-severity LVEF floors for pre-capillary PAH, so the
# prior v10 ramp (0.55→0.20) was invented.
#
# NOTE: CI is cardiac INDEX (L/min/m²), NOT absolute CO (L/min). The cost
# function computes CI = CO_lpm / BSA using the module-level DEFAULT_BSA_M2
# (overridable via --bsa CLI arg).

SEVERITY_PHYSIOLOGY = {
    # Low-risk band (Humbert T16 low-risk column)
    "healthy_low":     {"LV_EF_floor": 0.55, "RV_EF_floor": 0.54, "CI_min": 2.5},
    "healthy":         {"LV_EF_floor": 0.55, "RV_EF_floor": 0.54, "CI_min": 2.5},
    "borderline":      {"LV_EF_floor": 0.55, "RV_EF_floor": 0.54, "CI_min": 2.5},
    "mild":            {"LV_EF_floor": 0.55, "RV_EF_floor": 0.54, "CI_min": 2.5},
    # Intermediate band (Humbert T16 intermediate-risk column)
    "moderate":        {"LV_EF_floor": 0.55, "RV_EF_floor": 0.37, "CI_min": 2.0},
    "moderate_severe": {"LV_EF_floor": 0.55, "RV_EF_floor": 0.37, "CI_min": 2.0},
    # High-risk band (Humbert T16 high-risk column)
    "severe":          {"LV_EF_floor": 0.55, "RV_EF_floor": 0.25, "CI_min": 1.5},
    "very_severe":     {"LV_EF_floor": 0.55, "RV_EF_floor": 0.25, "CI_min": 1.5},
    "end_stage":       {"LV_EF_floor": 0.55, "RV_EF_floor": 0.25, "CI_min": 1.5},
}


# ==============================================================================
# PARAMETER CONFIG
#   (path, initial_guess, lower_bound, upper_bound, scale, note)
#   real_value = suggested_value * scale
# ==============================================================================
PARAMETER_CONFIG = [
    # -- Right heart -----------------------------------------------------------
    ("chambers.RV.EA",          0.8,    0.3,    2.5,    1.0,   "RV max elastance"),
    ("chambers.RV.EB",          0.06,   0.01,   0.25,   0.1,   "RV passive elastance"),
    ("chambers.RV.V0",          60.0,   10.0,   140.0,  1.0,   "RV unstressed volume [mL]"),
    ("chambers.RV.TC",          0.25,   0.18,   0.35,   1.0,   "RV contraction duration [s]"),
    ("chambers.RV.TR",          0.40,   0.28,   0.45,   1.0,   "RV relaxation duration [s]"),

    # -- Left heart ------------------------------------------------------------
    ("chambers.LV.EA",          2.8,    0.5,    15.0,   1.0,   "LV max elastance"),
    ("chambers.LV.EB",          0.09,   0.02,   0.30,   0.1,   "LV passive elastance"),
    ("chambers.LV.V0",          42.0,   20.0,   80.0,   1.0,   "LV unstressed volume [mL]"),
    ("chambers.LV.TC",          0.25,   0.18,   0.30,   1.0,   "LV contraction duration [s]"),
    ("chambers.LV.TR",          0.40,   0.28,   0.42,   1.0,   "LV relaxation duration [s]"),
    ("chambers.LA.EB",          0.05,   0.01,   0.15,   0.1,   "LA passive elastance"),
    ("chambers.LA.EA",          0.07,   0.02,   0.15,   0.1,   "LA active elastance"),

    # -- Valves ----------------------------------------------------------------
    ("valves.MV.Rmin",          0.01,   0.001,  0.08,   0.01,  "Mitral valve min resistance"),

    # -- Systemic circulation --------------------------------------------------
    ("circulation.SYS.R_AR",   0.90,   0.4,    1.4,    1.0,   "Systemic arterial resistance"),  # capped at clinical max ~1.4
    ("circulation.SYS.C_AR",   1.2,    0.4,    3.5,    1.0,   "Systemic arterial compliance"),
    ("circulation.SYS.C_VEN",  40.0,   10.0,   100.0,  10.0,  "Systemic venous compliance"),  # wider range for preload control

    # -- Pulmonary circulation -------------------------------------------------
    ("circulation.PUL.R_AR",   0.20,   0.05,   0.8,    0.1,   "Pulmonary arterial resistance"),
    ("circulation.PUL.C_AR",   0.50,   0.10,   1.0,    0.1,   "Pulmonary arterial compliance"),
    ("circulation.PUL.R_VEN",  0.15,   0.01,   1.0,    0.01,  "Pulmonary venous resistance"),
    ("circulation.PUL.C_VEN",  25.0,   5.0,    50.0,   10.0,  "Pulmonary venous compliance"),

    # -- Diastolic nonlinearity (exponential EDPVR) ----------------------------
    ("chambers.LV.kE",         0.01,   0.0,    0.04,   0.01,  "LV diastolic stiffening [1/mL]"),
    ("chambers.RV.kE",         0.01,   0.0,    0.04,   0.01,  "RV diastolic stiffening [1/mL]"),

    # -- Blood volume ----------------------------------------------------------
    ("TOTAL_VOLUME_OFFSET",    300.0,  0.0,    2000.0, 100.0, "Total volume offset [mL]"),
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


def _tolerances(pressure_targets, mesh_lv_edv, mesh_rv_edv):
    tol = {k: 15.0 for k in pressure_targets}
    tol["RV_EDP"] = 25.0
    tol["LV_EDP"] = 25.0
    tol["LV_ESP"] = 20.0
    tol["Ao_DBP"] = 20.0
    # Volume tolerances (added for reporting)
    tol["LV_EDV"] = 10.0
    tol["RV_EDV"] = 10.0
    tol["SV"] = 100.0  # emergent — always "pass"
    return tol


def _dynamic_weight(base_weight, rel_error, relax_threshold=0.05):
    """Reduce weight when close to target so optimizer focuses on remaining gaps."""
    if rel_error < relax_threshold:
        return base_weight * (rel_error / relax_threshold)
    return base_weight


def compute_cost(metrics, pressure_targets, mesh_lv_edv, mesh_rv_edv, rv_sv,
                  physiology=None, hr=1.25, bsa=DEFAULT_BSA_M2):
    """Cost: pressure targets + mesh volumes + SV balance + physiological guardrails.

    Pressure targets traced to primary literature; physiology floors are ESC
    risk-stratum bands (Humbert 2022 Table 16). See target_grounding_audit.md.
    """
    c = 0.0

    # --- PRESSURE targets ---
    # RV_ESP is the independent variable of the spectrum; LV_ESP, LV_EDP,
    # Ao_DBP, LA_P_MEAN are held at healthy-normal values across the spectrum.
    for key, base_w in [("RV_ESP", 200.0), ("LV_ESP", 200.0),
                        ("RV_EDP", 150.0), ("LV_EDP", 150.0),
                        ("Ao_DBP", 100.0), ("LA_P_MEAN", 150.0)]:
        rel = _rel(metrics[key], pressure_targets[key])
        c += _dynamic_weight(base_w, rel) * rel

    # --- VOLUME targets (match the mesh — makes ratio ~ 1) ---
    for edv, mesh in [("LV_EDV", mesh_lv_edv), ("RV_EDV", mesh_rv_edv)]:
        rel = _rel(metrics[edv], mesh)
        c += _dynamic_weight(400.0, rel) * rel

    # --- SV BALANCE (mass conservation — LV and RV must eject equally) ---
    c += 300.0 * abs(metrics["SV"] - rv_sv) / max(1.0, metrics["SV"])

    # --- PHYSIOLOGICAL GUARDRAILS (soft floor penalties from ESC bands) ---
    if physiology is not None:
        lv_edv = metrics["LV_EDV"]
        rv_edv = metrics["RV_EDV"]
        lv_sv = metrics["SV"]

        # LV EF floor — held at 0.55 for all severities (Baggen 2016: LVEF
        # not prognostic in PAH, so no per-severity ramp is justifiable).
        if lv_edv > 0:
            lv_ef = lv_sv / lv_edv
            ef_floor = physiology["LV_EF_floor"]
            if lv_ef < ef_floor:
                deficit = (ef_floor - lv_ef) / ef_floor
                c += 400.0 * deficit  # steep penalty below floor

        # RV EF floor — ESC band cutoffs from Humbert 2022 Table 16, p.45.
        if rv_edv > 0:
            rv_ef = rv_sv / rv_edv
            ef_floor_rv = physiology["RV_EF_floor"]
            if rv_ef < ef_floor_rv:
                deficit = (ef_floor_rv - rv_ef) / ef_floor_rv
                c += 300.0 * deficit

        # Cardiac INDEX barrier — Humbert 2022 Table 16 uses CI (L/min/m²)
        # not absolute CO. CI = CO_lpm / BSA.
        # Low-risk CI ≥2.5, intermediate 2.0-2.4, high <2.0 L/min/m².
        co_lpm = lv_sv * hr * 60.0 / 1000.0  # mL/beat * beats/s * 60 s/min / 1000
        ci = co_lpm / bsa
        ci_min = physiology["CI_min"]
        if ci < ci_min:
            deficit = (ci_min - ci) / ci_min
            c += 500.0 * deficit  # strong penalty — sub-physiological CI is unacceptable

    return c


# ==============================================================================
# MODEL INTERFACE
# ==============================================================================
class ModelInterface:
    def __init__(self, linear=False):
        """
        Parameters
        ----------
        linear : bool
            If True, force kE=0 for both ventricles (linear Regazzoni EDPVR).
            If False (default), kE is a free parameter in PARAMETER_CONFIG
            and the optimizer chooses it (exponential Klotz-style EDPVR).
        """
        self.linear = linear
        base = Regazzoni2020(add_units=False)
        self.base_params = base.parameters
        self.base_init = base._initial_state

    def build(self, suggested: dict):
        params = self.base_params.copy()
        init_state = self.base_init.copy()

        for key, _, _, _, scale, _ in PARAMETER_CONFIG:
            # Skip kE entries when --linear is set; they are set to 0 below.
            if self.linear and key in ("chambers.LV.kE", "chambers.RV.kE"):
                continue
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

        # Force linear EDPVR by pinning kE to exactly zero if requested.
        if self.linear:
            params["chambers"]["LV"]["kE"] = 0.0
            params["chambers"]["RV"]["kE"] = 0.0

        return params, init_state


# ==============================================================================
# OBJECTIVE FACTORY
# ==============================================================================
def make_objective(interface, pressure_targets, mesh_lv_edv, mesh_rv_edv,
                   physiology=None, bsa=DEFAULT_BSA_M2):
    # Clamp V0 upper bounds: unstressed volume must be below EDV
    v0_caps = {
        "chambers.RV.V0": mesh_rv_edv - 5.0,
        "chambers.LV.V0": mesh_lv_edv - 5.0,
    }
    # In --linear mode, kE is fixed at 0 and excluded from the Optuna search.
    kE_keys = ("chambers.LV.kE", "chambers.RV.kE")

    def objective(trial):
        suggested = {}
        for key, _, lb, ub, scale, _ in PARAMETER_CONFIG:
            if interface.linear and key in kE_keys:
                continue  # not searched; interface.build pins these to 0
            lo, hi = lb / scale, ub / scale
            if key in v0_caps:
                hi = min(hi, v0_caps[key] / scale)
                lo = min(lo, hi - 1.0)  # ensure lo < hi
            suggested[key] = trial.suggest_float(key, lo, hi)

        params, init_state = interface.build(suggested)

        # Guard: activation timing must fit within one heartbeat
        # Return graded penalty instead of pruning (gives optimizer gradient info)
        lv = params["chambers"]["LV"]
        timing = lv["TC"] + lv["TR"] + lv.get("tC", 0.1)
        if timing > 0.78:
            return 5000.0 + 1000.0 * (timing - 0.78)

        model = Regazzoni2020(
            parameters=params, initial_state=init_state,
            add_units=False, verbose=False,
        )
        try:
            history = model.solve(num_beats=20, dt=1e-3)
        except Exception:
            return 8000.0

        hr = params["HR"]
        samples = int((1 / hr) / 1e-3)

        # Graded penalties instead of hard pruning
        prv_max = np.max(history["p_RV"][-samples:])
        if np.isnan(prv_max):
            return 9000.0
        if prv_max > 300.0:
            return 5000.0 + prv_max

        # Extract metrics from last two beats
        slc_last = slice(-samples, None)
        slc_prev = slice(-2 * samples, -samples)

        p_lv, v_lv = history["p_LV"][slc_last], history["V_LV"][slc_last]
        p_rv, v_rv = history["p_RV"][slc_last], history["V_RV"][slc_last]
        p_ao = history["p_AR_SYS"][slc_last]
        p_la = history["p_LA"][slc_last]

        metrics = extract_metrics(p_lv, v_lv, p_rv, v_rv, p_ao, p_la)
        rv_sv = np.max(v_rv) - np.min(v_rv)

        cost = compute_cost(
            metrics, pressure_targets, mesh_lv_edv, mesh_rv_edv, rv_sv,
            physiology=physiology, hr=hr, bsa=bsa,
        )

        # Convergence penalty: compare SV between last two beats
        sv_last = metrics["SV"]
        sv_prev = np.max(history["V_LV"][slc_prev]) - np.min(history["V_LV"][slc_prev])
        rv_sv_prev = np.max(history["V_RV"][slc_prev]) - np.min(history["V_RV"][slc_prev])
        sv_drift = abs(sv_last - sv_prev) + abs(rv_sv - rv_sv_prev)
        cost += 200.0 * sv_drift / max(1.0, sv_last)

        return cost

    return objective


# ==============================================================================
# OPTIMISATION
# ==============================================================================
def run_optimization(severity, mesh_lv_edv, mesh_rv_edv, mesh_name,
                     n_trials, outdir, seed=42, bsa=DEFAULT_BSA_M2,
                     linear=False):
    pressure_targets = SEVERITY_PRESSURE_TARGETS[severity]
    physiology = SEVERITY_PHYSIOLOGY.get(severity)
    edpvr_tag = "linear" if linear else "exp"
    study_name = f"v11_mesh_{mesh_name}_{severity}_{edpvr_tag}"
    out_json = str(outdir / f"optimized_regazzoni_{mesh_name}_{severity}.json")

    edpvr_desc = "linear (kE pinned at 0)" if linear else "exponential Klotz (kE free)"
    print("\n" + "=" * 65)
    print(f"  Optimising: {severity} ({mesh_name} mesh, {n_trials} trials)")
    print(f"  Mesh volumes: LV_EDV={mesh_lv_edv:.1f} mL, RV_EDV={mesh_rv_edv:.1f} mL")
    print(f"  BSA: {bsa:.2f} m²")
    print(f"  EDPVR: {edpvr_desc}")
    print("=" * 65)
    print("\nPressure targets:")
    for k, v in pressure_targets.items():
        print(f"  {k:<14}: {v}")
    print(f"\nVolume targets (from mesh):")
    print(f"  LV_EDV       : {mesh_lv_edv:.1f}")
    print(f"  RV_EDV       : {mesh_rv_edv:.1f}")
    print(f"  SV           : (emergent)")
    if physiology:
        print(f"\nPhysiological guardrails (ESC Humbert 2022 T16 bands):")
        print(f"  LV EF floor  : {physiology['LV_EF_floor']*100:.0f}%")
        print(f"  RV EF floor  : {physiology['RV_EF_floor']*100:.0f}%")
        print(f"  CI minimum   : {physiology['CI_min']:.1f} L/min/m²")

    interface = ModelInterface(linear=linear)

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{outdir / f'{study_name}.db'}",
        load_if_exists=True,
        direction="minimize",
        sampler=CmaEsSampler(seed=seed, warn_independent_sampling=False),
    )

    study.optimize(
        make_objective(interface, pressure_targets, mesh_lv_edv, mesh_rv_edv,
                       physiology=physiology, bsa=bsa),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    print(f"\n  Best cost: {study.best_value:.4f}")

    # Reconstruct best parameters and verify at high resolution
    params, init_state = interface.build(study.best_params)
    model = Regazzoni2020(
        parameters=params, initial_state=init_state,
        add_units=False, verbose=False,
    )
    history = model.solve(num_beats=50, dt=1e-3)

    samples = int((1 / params["HR"]) / 1e-3)
    slc = slice(-samples, None)
    metrics = extract_metrics(
        history["p_LV"][slc], history["V_LV"][slc],
        history["p_RV"][slc], history["V_RV"][slc],
        history["p_AR_SYS"][slc], history["p_LA"][slc],
    )

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Compute derived hemodynamic metrics
    hr = params["HR"]
    lv_ef = metrics["SV"] / metrics["LV_EDV"] * 100 if metrics["LV_EDV"] > 0 else 0
    rv_edv_val = metrics["RV_EDV"]
    rv_sv_val = np.max(history["V_RV"][slc]) - np.min(history["V_RV"][slc])
    rv_ef = rv_sv_val / rv_edv_val * 100 if rv_edv_val > 0 else 0
    co_lpm = metrics["SV"] * hr * 60.0 / 1000.0
    ci_lpm_m2 = co_lpm / bsa

    save_data = {
        "description": f"Mesh-specific circulation: {mesh_name} {severity} "
                       f"({'linear' if linear else 'exponential'} EDPVR)",
        "mesh_volumes_mL": {"LV_EDV": mesh_lv_edv, "RV_EDV": mesh_rv_edv},
        "severity": severity,
        "bsa_m2": bsa,
        "edpvr_model": "linear" if linear else "exponential_klotz",
        "pressure_targets": pressure_targets,
        "physiology_guardrails": physiology,
        "metrics_achieved": metrics,
        "derived_hemodynamics": {
            "LV_EF_pct": lv_ef, "RV_EF_pct": rv_ef,
            "CO_Lpm": co_lpm, "CI_Lpm_m2": ci_lpm_m2, "RV_SV": rv_sv_val,
        },
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
def evaluate_and_plot(severity, mesh_lv_edv, mesh_rv_edv, mesh_name, json_path, outdir,
                       bsa=DEFAULT_BSA_M2):
    pressure_targets = SEVERITY_PRESSURE_TARGETS[severity]
    tols = _tolerances(pressure_targets, mesh_lv_edv, mesh_rv_edv)

    print(f"\nEvaluating {json_path} ({severity}, {mesh_name} mesh) ...")

    with open(json_path) as f:
        data = json.load(f)

    parameters = data["parameters"]
    initial_state = data["initial_state"]

    model = Regazzoni2020(
        parameters=parameters, initial_state=initial_state,
        add_units=False, verbose=False,
    )
    history = model.solve(num_beats=50, dt=1e-3)

    samples = int((1 / parameters["HR"]) / 1e-3)
    slc = slice(-samples, None)

    v_lv = history["V_LV"][slc]; p_lv = history["p_LV"][slc]
    v_rv = history["V_RV"][slc]; p_rv = history["p_RV"][slc]
    p_ao = history["p_AR_SYS"][slc]
    p_la = history["p_LA"][slc]
    t = np.linspace(0, 1 / parameters["HR"], samples)

    m = extract_metrics(p_lv, v_lv, p_rv, v_rv, p_ao, p_la)

    # -- Build combined targets for reporting ----------------------------------
    report_targets = dict(pressure_targets)
    report_targets["LV_EDV"] = mesh_lv_edv
    report_targets["RV_EDV"] = mesh_rv_edv
    report_targets["SV"] = m["SV"]  # emergent — report but always "pass"

    # -- Console table ---------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'METRIC':<14} | {'TARGET':>8} | {'ACHIEVED':>9} | "
          f"{'ERROR':>8} | {'TOL':>5} | STATUS")
    print("-" * 72)
    all_pass = True
    for key, tgt in report_targets.items():
        val = m[key]
        if key == "SV":
            print(f"{'SV (emergent)':<14} | {'--':>8} | {val:>9.2f} | "
                  f"{'--':>8} | {'--':>5} | INFO")
            continue
        err = (val - tgt) / tgt * 100
        tol = tols.get(key, 15.0)
        ok = abs(err) <= tol
        if not ok:
            all_pass = False
        flag = "PASS" if ok else "FAIL"
        print(f"{key:<14} | {tgt:>8.1f} | {val:>9.2f} | "
              f"{err:>+7.1f}% | +/-{tol:>3.0f}% | {flag}")

    # SV balance diagnostic
    lv_sv = m["SV"]
    rv_sv = np.max(v_rv) - np.min(v_rv)
    sv_diff = abs(lv_sv - rv_sv)
    sv_ok = sv_diff < 2.0
    print(f"{'SV_BALANCE':<14} | {'0.0':>8} | {sv_diff:>9.2f} | "
          f"{'':>8} | {'<2mL':>5} | {'PASS' if sv_ok else 'FAIL'}")
    if not sv_ok:
        all_pass = False

    # EF and CI diagnostics
    lv_ef = (lv_sv / m["LV_EDV"] * 100) if m["LV_EDV"] > 0 else 0
    rv_ef = (rv_sv / m["RV_EDV"] * 100) if m["RV_EDV"] > 0 else 0
    co_lpm = lv_sv * parameters["HR"] * 60.0 / 1000.0
    ci = co_lpm / bsa
    physiology = SEVERITY_PHYSIOLOGY.get(severity, {})
    lv_ef_floor = physiology.get("LV_EF_floor", 0.30) * 100
    rv_ef_floor = physiology.get("RV_EF_floor", 0.20) * 100
    ci_min = physiology.get("CI_min", 1.5)
    lv_ef_ok = lv_ef >= lv_ef_floor
    rv_ef_ok = rv_ef >= rv_ef_floor
    ci_ok = ci >= ci_min
    print(f"{'LV_EF':<14} | {lv_ef_floor:>7.0f}% | {lv_ef:>8.1f}% | "
          f"{'':>8} | {'floor':>5} | {'PASS' if lv_ef_ok else 'FAIL'}")
    print(f"{'RV_EF':<14} | {rv_ef_floor:>7.0f}% | {rv_ef:>8.1f}% | "
          f"{'':>8} | {'floor':>5} | {'PASS' if rv_ef_ok else 'FAIL'}")
    print(f"{'CI (L/min/m²)':<14} | {ci_min:>8.1f} | {ci:>9.2f} | "
          f"{'':>8} | {'floor':>5} | {'PASS' if ci_ok else 'FAIL'}")
    if not lv_ef_ok or not rv_ef_ok or not ci_ok:
        all_pass = False

    # Volume ratio diagnostic
    ratio_lv = mesh_lv_edv / m["LV_EDV"] if m["LV_EDV"] > 0 else 0
    ratio_rv = mesh_rv_edv / m["RV_EDV"] if m["RV_EDV"] > 0 else 0
    print(f"{'ratio_LV':<14} | {'1.00':>8} | {ratio_lv:>9.3f} | "
          f"{'':>8} | {'':>5} | INFO")
    print(f"{'ratio_RV':<14} | {'1.00':>8} | {ratio_rv:>9.3f} | "
          f"{'':>8} | {'':>5} | INFO")

    print("-" * 72)
    print(f"Overall: {'ALL METRICS PASS' if all_pass else 'SOME METRICS FAILED'}")
    print("=" * 72)

    # -- Figure (2x2) ----------------------------------------------------------
    label = severity.replace("_", " ").title()

    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    ax_lv  = fig.add_subplot(gs[0, 0])
    ax_rv  = fig.add_subplot(gs[0, 1])
    ax_ao  = fig.add_subplot(gs[1, 0])
    ax_tbl = fig.add_subplot(gs[1, 1])

    # LV PV loop
    ax_lv.plot(v_lv, p_lv, "r-", lw=2.5)
    ax_lv.fill(v_lv, p_lv, alpha=0.12, color="red")
    ax_lv.axhline(pressure_targets["LV_ESP"], color="darkred", ls="--", lw=1.2,
                  label=f"Target ESP {pressure_targets['LV_ESP']:.0f}")
    ax_lv.axvline(mesh_lv_edv, color="tomato", ls="--", lw=1.2,
                  label=f"Mesh EDV {mesh_lv_edv:.0f}")
    ax_lv.set_title(f"LV PV Loop ({label})\n"
                    f"ESP {m['LV_ESP']:.1f}  EDV {m['LV_EDV']:.1f}  "
                    f"SV {m['SV']:.1f}", fontsize=9)
    ax_lv.set_xlabel("Volume [mL]")
    ax_lv.set_ylabel("Pressure [mmHg]")
    ax_lv.legend(fontsize=7.5)
    ax_lv.grid(True, alpha=0.3)

    # RV PV loop
    ax_rv.plot(v_rv, p_rv, "b-", lw=2.5)
    ax_rv.fill(v_rv, p_rv, alpha=0.12, color="blue")
    ax_rv.axhline(pressure_targets["RV_ESP"], color="navy", ls="--", lw=1.2,
                  label=f"Target ESP {pressure_targets['RV_ESP']:.0f}")
    ax_rv.axhline(pressure_targets["RV_EDP"], color="steelblue", ls=":", lw=1.2,
                  label=f"Target EDP {pressure_targets['RV_EDP']:.0f}")
    ax_rv.axvline(mesh_rv_edv, color="cornflowerblue", ls="--", lw=1.2,
                  label=f"Mesh EDV {mesh_rv_edv:.0f}")
    ax_rv.set_title(f"RV PV Loop ({label})\n"
                    f"ESP {m['RV_ESP']:.1f}  EDP {m['RV_EDP']:.1f}  "
                    f"EDV {m['RV_EDV']:.1f}", fontsize=9)
    ax_rv.set_xlabel("Volume [mL]")
    ax_rv.set_ylabel("Pressure [mmHg]")
    ax_rv.legend(fontsize=7.5)
    ax_rv.grid(True, alpha=0.3)

    # Aortic + LA traces
    ax_ao.plot(t, p_ao, "k-", lw=2, label=f"Ao (DBP {m['Ao_DBP']:.1f})")
    ax_ao.plot(t, p_la, "m--", lw=1.5, label=f"LA (mean {m['LA_P_MEAN']:.1f})")
    ax_ao.axhline(pressure_targets["Ao_DBP"], color="gray", ls="--", lw=1)
    ax_ao.axhline(pressure_targets["LA_P_MEAN"], color="violet", ls=":", lw=1)
    ax_ao.set_title(f"Aortic & LA Pressure\n"
                    f"Ao_DBP {m['Ao_DBP']:.1f}  LA_mean {m['LA_P_MEAN']:.1f}",
                    fontsize=9)
    ax_ao.set_xlabel("Time [s]")
    ax_ao.set_ylabel("Pressure [mmHg]")
    ax_ao.legend(fontsize=7.5)
    ax_ao.grid(True, alpha=0.3)

    # Metrics table
    ax_tbl.axis("off")
    col_labels = ["Metric", "Target", "Achieved", "Error %", "Status"]
    rows, cell_colors = [], []

    for key in list(pressure_targets.keys()) + ["LV_EDV", "RV_EDV"]:
        val = m[key]
        if key in pressure_targets:
            tgt = pressure_targets[key]
        elif key == "LV_EDV":
            tgt = mesh_lv_edv
        else:
            tgt = mesh_rv_edv
        err = (val - tgt) / tgt * 100
        tol = tols.get(key, 15.0)
        ok = abs(err) <= tol
        rows.append([key, f"{tgt:.1f}", f"{val:.2f}", f"{err:+.1f}%",
                     "PASS" if ok else "FAIL"])
        cell_colors.append(
            ["#f5f5f5"] * 4 + ["#c6efce" if ok else "#ffc7ce"]
        )

    # Add SV row (emergent)
    rows.append(["SV (emergent)", "--", f"{m['SV']:.1f}", "--", "INFO"])
    cell_colors.append(["#f5f5f5"] * 4 + ["#daeef3"])

    # Add EF and CO rows
    rows.append(["LV EF", f">{lv_ef_floor:.0f}%", f"{lv_ef:.1f}%", "--",
                 "PASS" if lv_ef_ok else "FAIL"])
    cell_colors.append(["#f5f5f5"] * 4 + ["#c6efce" if lv_ef_ok else "#ffc7ce"])
    rows.append(["RV EF", f">{rv_ef_floor:.0f}%", f"{rv_ef:.1f}%", "--",
                 "PASS" if rv_ef_ok else "FAIL"])
    cell_colors.append(["#f5f5f5"] * 4 + ["#c6efce" if rv_ef_ok else "#ffc7ce"])
    rows.append(["CI (L/min/m²)", f">{ci_min:.1f}", f"{ci:.2f}", "--",
                 "PASS" if ci_ok else "FAIL"])
    cell_colors.append(["#f5f5f5"] * 4 + ["#c6efce" if ci_ok else "#ffc7ce"])

    # Add SV balance row
    sv_flag = "PASS" if sv_ok else "FAIL"
    rows.append(["SV balance", "0.0", f"{sv_diff:.2f}", "--", sv_flag])
    cell_colors.append(
        ["#f5f5f5"] * 4 + ["#c6efce" if sv_ok else "#ffc7ce"]
    )

    tbl = ax_tbl.table(
        cellText=rows, colLabels=col_labels,
        cellColours=cell_colors, loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472c4")
            cell.set_text_props(color="white", fontweight="bold")

    ax_tbl.set_title(f"{label} — Targets vs Achieved\n"
                     f"(mesh: {mesh_name}, ratio_LV={ratio_lv:.3f}, "
                     f"ratio_RV={ratio_rv:.3f})",
                     fontsize=10, fontweight="bold", pad=8)

    out_fig = str(outdir / f"pv_evaluation_{mesh_name}_{severity}.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_fig}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    all_severities = list(SEVERITY_PRESSURE_TARGETS.keys())

    parser = argparse.ArgumentParser(
        description="Mesh-specific 0D circulation optimizer (Option D).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Severities: {', '.join(all_severities)}, all",
    )
    parser.add_argument(
        "--mesh", default=None,
        help="Path to XDMF mesh file, cardiac_geometries folder, or 'ukb' "
             "to generate the synthetic UK Biobank mesh.  Cavity volumes "
             "are read directly from the mesh.",
    )
    parser.add_argument(
        "--lv-edv", type=float, default=None,
        help="Override LV end-diastolic volume [mL] (skip mesh reading).",
    )
    parser.add_argument(
        "--rv-edv", type=float, default=None,
        help="Override RV end-diastolic volume [mL] (skip mesh reading).",
    )
    parser.add_argument(
        "--severity", required=True,
        choices=all_severities + ["all"],
        help="Disease severity level (or 'all').",
    )
    parser.add_argument(
        "--mesh-unit", choices=["mm", "cm", "m"], default=None,
        help="Coordinate unit of the mesh (mm, cm, m).  Determines the "
             "volume→mL conversion factor.  Default: mm for ukb/directory, "
             "cm for XDMF files.",
    )
    parser.add_argument(
        "--mesh-name", default=None,
        help="Override mesh identifier for output filenames "
             "(default: derived from mesh path).",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip optimisation; evaluate saved JSON only.",
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
    parser.add_argument(
        "--bsa", type=float, default=DEFAULT_BSA_M2,
        help=f"Body surface area in m² for CI calculation "
             f"(default: {DEFAULT_BSA_M2}, UKB mean mesh).",
    )
    parser.add_argument(
        "--linear", action="store_true",
        help="Force linear Regazzoni EDPVR by pinning kE=0 for both ventricles. "
             "Default is exponential Klotz-style EDPVR (kE free parameter). "
             "When --linear is set, the mesh-name gets a '_linear' suffix in "
             "output filenames so A/B test results don't collide.",
    )
    args = parser.parse_args()

    # Read cavity volumes: from mesh, or from CLI overrides
    if args.lv_edv is not None and args.rv_edv is not None:
        lv_edv, rv_edv = args.lv_edv, args.rv_edv
        auto_name = args.mesh_name or "custom"
    elif args.mesh is not None:
        lv_edv, rv_edv, auto_name = read_mesh_volumes(args.mesh, mesh_unit=args.mesh_unit)
    else:
        parser.error("Provide --mesh, or both --lv-edv and --rv-edv.")
    mesh_name = args.mesh_name or auto_name
    # When --linear is active, tag the mesh-name so output directories, JSON
    # files, and Optuna databases do not collide with the exponential run.
    if args.linear and not mesh_name.endswith("_linear"):
        mesh_name = f"{mesh_name}_linear"

    # Create output directory next to this script
    outdir = Path(__file__).resolve().parent / f"results_mesh_{mesh_name}"
    outdir.mkdir(exist_ok=True)

    print(f"\nMesh volumes: LV_EDV = {lv_edv:.1f} mL, RV_EDV = {rv_edv:.1f} mL")
    print(f"BSA: {args.bsa:.2f} m²")
    print(f"EDPVR model: {'LINEAR (kE=0 fixed)' if args.linear else 'EXPONENTIAL (kE free)'}")
    print(f"Output directory: {outdir}")

    severities = all_severities if args.severity == "all" else [args.severity]

    for sev in severities:
        out_json = args.json or str(
            outdir / f"optimized_regazzoni_{mesh_name}_{sev}.json")

        if not args.eval_only:
            out_json = run_optimization(
                sev, lv_edv, rv_edv, mesh_name,
                args.n_trials, outdir, args.seed, bsa=args.bsa,
                linear=args.linear,
            )

        evaluate_and_plot(sev, lv_edv, rv_edv, mesh_name, out_json, outdir,
                          bsa=args.bsa)


if __name__ == "__main__":
    main()

# %% [markdown]
# # Optimizing for Pulmonary Hypertension (Disease Modeling)
#
# This script evolves a Healthy Baseline into a **Severe Pulmonary Hypertension (PH)** state.
#
# ## Critical Constraints for "Pre-Capillary" PH
# To ensure we simulate **Group 1 PH** (Vascular Disease) and not Heart Failure:
# 1. **LA Pressure must be NORMAL:** High Pulmonary pressure with Normal LA pressure confirms the resistance is in the lungs.
# 2. **RV Dilation:** The RV will dilate, but we target a specific remodeling range (~190 mL) to avoid unphysiological ballooning.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from circulation.regazzoni2020 import Regazzoni2020
import logging

# Define Targets
# Sources:
# 1. Definition of Pre-Capillary PH (mPAP > 20, PAWP <= 15):
#    Humbert et al., "2022 ESC/ERS Guidelines...", Eur Heart J (2022).
# 2. Reduced Stroke Volume/Cardiac Index in Severe PH:
#    D'Alonzo et al., "Survival in patients with primary pulmonary hypertension", Ann Intern Med (1991).
# 3. Ventricular Interdependence (LV underfilling):
#    Vonk Noordegraaf et al., "The right ventricle in pulmonary hypertension", Circ Res (2017).

targets = {
    # --- The Cause (Pulmonary Vascular Disease) ---
    "RV_ESP": 60.0,   # High Pulmonary Pressure (Severe PH > 20 mean)
    "RV_EDP": 10.0,   # Elevated RV filling pressure

    # --- The Constraint (Definition of Pre-Capillary) ---
    "LA_P_MEAN": 9.0, # Normal (< 15 mmHg). This proves the LV is healthy (Group 1 PH).

    # --- The Consequence (Systemic Effects) ---
    "SV":     55.0,   # Reduced due to blockage in lungs (Forward Failure)
    "LV_ESP": 110.0,  # Lower systemic pressure due to lower SV
    "Ao_DBP": 75.0,   # Lower diastolic pressure

    # --- The Remodeling (Anatomy) ---
    "RV_EDV": 190.0,  # Dilated RV (Adaptive mechanism to maintain SV)
}

print("Target PH Hemodynamics:")
for key, value in targets.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## Configuration
# We use the Healthy Baseline as the scaling factor (1.0).

# %%
class PHModelInterface:
    def __init__(self):
        self.base_model = Regazzoni2020(add_units=False)
        self.base_params = self.base_model.parameters
        self.base_init = self.base_model._initial_state

        self.config = [
            # 1. Pulmonary Resistance (The Root Cause)
            ("circulation.PUL.R_AR",    5.0,    1.0,   20.0,  self.base_params["circulation"]["PUL"]["R_AR"]),

            # 2. Pulmonary Compliance (Stiffening)
            ("circulation.PUL.C_AR",    0.5,    0.1,   1.0,   self.base_params["circulation"]["PUL"]["C_AR"]),

            # 3. RV Contractility (Remodeling: Hypertrophy)
            ("chambers.RV.EA",          2.0,    1.0,   10.0,  self.base_params["chambers"]["RV"]["EA"]),

            # 4. RV Stiffness (Remodeling: Fibrosis)
            ("chambers.RV.EB",          2.0,    1.0,   10.0,  self.base_params["chambers"]["RV"]["EB"]),

            # 5. Systemic Compensation
            ("circulation.SYS.R_AR",    1.1,    0.8,   2.0,   self.base_params["circulation"]["SYS"]["R_AR"]),

            # 6. Volume Status (Fluid Retention)
            ("TOTAL_VOLUME_OFFSET",     0.0,   -200.0, 1000.0, 100.0)
        ]

    def get_initial_guess(self):
        return [val for _, val, _, _, _ in self.config]

    def get_bounds(self):
        return [(lb, ub) for _, _, lb, ub, _ in self.config]

    def update_model(self, scaled_x):
        params = self.base_params.copy()
        init_state = self.base_init.copy()

        for val, (key, _, _, _, scale) in zip(scaled_x, self.config):
            real_val = val * scale
            if key == "TOTAL_VOLUME_OFFSET":
                C_ven = params["circulation"]["SYS"]["C_VEN"]
                current_p = init_state["p_VEN_SYS"]
                if hasattr(current_p, "magnitude"): current_p = current_p.magnitude
                init_state["p_VEN_SYS"] = current_p + (real_val / C_ven)
            else:
                keys = key.split(".")
                d = params
                for k in keys[:-1]: d = d[k]
                d[keys[-1]] = real_val
        return params, init_state

# %% [markdown]
# ## The Cost Function (With Traffic Guards)

# %%
interface = PHModelInterface()
iteration_counter = [0]

def cost_function(scaled_x):
    params, init_state = interface.update_model(scaled_x)
    logging.getLogger('circulation.base').setLevel(logging.WARNING)
    model = Regazzoni2020(parameters=params, initial_state=init_state, add_units=False, verbose=False)

    try:
        history = model.solve(num_beats=10, dt=2e-3)
    except (RuntimeError, ValueError):
        return 1e6

    samples = int((1/params["HR"]) / 2e-3)
    slc = slice(-samples, None)

    p_rv = history["p_RV"][slc]
    v_rv = history["V_RV"][slc]
    p_lv = history["p_LV"][slc]
    p_la = history["p_LA"][slc]  # Need LA Pressure
    p_ao = history["p_AR_SYS"][slc] # Need Ao Pressure

    if np.max(p_rv) > 300.0 or np.isnan(np.sum(p_rv)): return 1e6

    metrics = {
        "RV_ESP": np.max(p_rv),
        "RV_EDP": np.min(p_rv),
        "SV":     np.max(v_rv) - np.min(v_rv),
        "LV_ESP": np.max(p_lv),
        "LA_P_MEAN": np.mean(p_la),
        "RV_EDV": np.max(v_rv),
        "Ao_DBP": np.min(p_ao)
    }

    cost = 0.0

    # 1. Primary Disease Targets (High RV Pressure)
    cost += 40.0 * ((metrics["RV_ESP"] - targets["RV_ESP"]) / targets["RV_ESP"])**2
    cost += 20.0 * ((metrics["RV_EDP"] - targets["RV_EDP"]) / targets["RV_EDP"])**2

    # 2. Cardiac Output Target
    cost += 30.0 * ((metrics["SV"] - targets["SV"]) / targets["SV"])**2

    # 3. CRITICAL: The "Anti-Cheat" Constraints
    # High LA Pressure = Wrong Disease (Left Heart Failure). Heavy Penalty.
    cost += 50.0 * ((metrics["LA_P_MEAN"] - targets["LA_P_MEAN"]) / targets["LA_P_MEAN"])**2

    # Control RV Dilation (Allow some, but penalize excessive ballooning)
    cost += 15.0 * ((metrics["RV_EDV"] - targets["RV_EDV"]) / targets["RV_EDV"])**2

    # 4. Systemic BP Constraints
    cost += 10.0 * ((metrics["Ao_DBP"] - targets["Ao_DBP"]) / targets["Ao_DBP"])**2
    cost += 10.0 * ((metrics["LV_ESP"] - targets["LV_ESP"]) / targets["LV_ESP"])**2

    if iteration_counter[0] % 10 == 0:
        print(f"Iter {iteration_counter[0]:3d} | Cost: {cost:.4f} | RV P: {metrics['RV_ESP']:.0f} | LA P: {metrics['LA_P_MEAN']:.1f}")

    iteration_counter[0] += 1
    return cost

# %%
print("\nStarting Optimization...")
x0 = interface.get_initial_guess()
bounds = interface.get_bounds()
iteration_counter[0] = 0
result = minimize(cost_function, x0, method='Nelder-Mead',
                  bounds=Bounds([b[0] for b in bounds], [b[1] for b in bounds]),
                  options={'maxiter': 20 if os.getenv("CI") else 1000, 'xatol': 1e-4, 'disp': True})

# %% [markdown]
# ## Traffic Light Report

# %%
ph_params, ph_init = interface.update_model(result.x)
print("\n" + "="*60)
print("Running Verification Simulation...")
ph_model = Regazzoni2020(parameters=ph_params, initial_state=ph_init, add_units=False, verbose=False)
ph_hist = ph_model.solve(num_beats=20, dt=1e-3)

samples = int((1/ph_params["HR"]) / 1e-3)
slc = slice(-samples, None)

p_rv = ph_hist["p_RV"][slc]
v_rv = ph_hist["V_RV"][slc]
p_lv = ph_hist["p_LV"][slc]
v_lv = ph_hist["V_LV"][slc]
p_la = ph_hist["p_LA"][slc]
p_ao = ph_hist["p_AR_SYS"][slc]

achieved = {
    "RV_ESP": np.max(p_rv),
    "RV_EDP": np.min(p_rv),
    "SV":     np.max(v_rv) - np.min(v_rv),
    "LV_ESP": np.max(p_lv),
    "LA_P_MEAN": np.mean(p_la),
    "RV_EDV": np.max(v_rv),
    "Ao_DBP": np.min(p_ao)
}

print("\n" + "="*60)
print(f"{'METRIC':<15} | {'TARGET':<10} | {'ACHIEVED':<10} | {'ERROR':<8} | {'STATUS'}")
print("-" * 60)

for key, target in targets.items():
    val = achieved.get(key, 0.0)
    error_pct = ((val - target) / target) * 100

    # Strict tolerance for LA Pressure (to avoid misdiagnosis)
    tol = 15.0 if key != "LA_P_MEAN" else 10.0

    status = "[PASS]" if abs(error_pct) <= tol else "[FAIL]"
    print(f"{key:<15} | {target:<10.1f} | {val:<10.1f} | {error_pct:>+6.1f}% | {status}")
print("-" * 60)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# RV Plot
axs[0].plot(v_rv, p_rv, 'r-', lw=3, label="PH (Optimized)")
axs[0].set_title(f"Right Ventricle (Target ESP: {targets['RV_ESP']})")
axs[0].axhline(targets['RV_ESP'], color='k', ls=':', alpha=0.5)
axs[0].axvline(targets['RV_EDV'], color='r', ls='--', label=f"Max EDV ({targets['RV_EDV']})")
axs[0].set_xlabel("Volume [mL]")
axs[0].set_ylabel("Pressure [mmHg]")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# LA Pressure Plot (The Proof of Pre-Capillary PH)
axs[1].plot(ph_hist["time"][slc], p_la, 'g-', lw=2, label="LA Pressure")
axs[1].axhline(targets['LA_P_MEAN'], color='k', ls='--', label="Target Mean")
axs[1].set_title(f"Left Atrial Pressure (Target: {targets['LA_P_MEAN']})")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Pressure [mmHg]")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#
# Notice how the Right Ventricle loop shoots upward, indicating massive pressure generation. Conversely, the Left Ventricle loop shrinks slightly and shifts left, indicating it is under-filled and pumping less blood, despite being "healthy" itself.

# %%
# 1. Get Optimized PH Model
ph_params, ph_init = interface.update_model(result.x)
ph_model = Regazzoni2020(parameters=ph_params, initial_state=ph_init, add_units=False)
logging.getLogger('circulation.base').setLevel(logging.WARNING)
ph_hist = ph_model.solve(num_beats=20, dt=1e-3)

# 2. Get Healthy Baseline Model
healthy_model = Regazzoni2020(add_units=False) # Uses defaults
healthy_hist = healthy_model.solve(num_beats=20, dt=1e-3)

# 3. Plot
samples = int((1/ph_params["HR"]) / 1e-3)
slc = slice(-samples, None)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# RV Loop (The main event)
axs[0].plot(healthy_hist["V_RV"][slc], healthy_hist["p_RV"][slc], 'g--', lw=2, label="Healthy Baseline")
axs[0].plot(ph_hist["V_RV"][slc], ph_hist["p_RV"][slc], 'r-', lw=3, label="Pulmonary Hypertension")
axs[0].set_title("Right Ventricle (Disease State)")
axs[0].set_xlabel("Volume [mL]")
axs[0].set_ylabel("Pressure [mmHg]")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# LV Loop (The consequence)
axs[1].plot(healthy_hist["V_LV"][slc], healthy_hist["p_LV"][slc], 'g--', lw=2, label="Healthy Baseline")
axs[1].plot(ph_hist["V_LV"][slc], ph_hist["p_LV"][slc], 'k-', lw=2, label="PH (Under-filled)")
axs[1].set_title("Left Ventricle (Secondary Effect)")
axs[1].set_xlabel("Volume [mL]")
axs[1].set_ylabel("Pressure [mmHg]")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Parameter Changes
# Finally, let's look at the "Remodeling Factor." This table shows exactly how much the parameters had to change from the healthy baseline to create this disease state. A factor of `1.0x` means no change; `5.0x` means a 5-fold increase.

# %%
print("\nParameter Remodeling (Factor of Healthy Baseline):")
print(f"{'Parameter':<25} | {'Healthy':<10} | {'Disease':<10} | {'Factor':<10}")
print("-" * 65)

for i, (key, _, _, _, scale) in enumerate(interface.config):
    if "OFFSET" in key: continue

    new_val = result.x[i] * scale
    factor = new_val / scale
    print(f"{key:<25} | {scale:<10.4f} | {new_val:<10.4f} | {factor:<10.2f}x")

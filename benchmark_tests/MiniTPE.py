import os
import time
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# (A) Path to the DataCube CSV (semicolon-delimited)
DATA_CUBE_FILE = r"../DataCube.csv"

# (B) Integer domain bounds for Phosphate and Temperature
PHOSPHATE_BOUNDS = (1, 20)
TEMPERATURE_BOUNDS = (4, 36)

# (C) Hard cap on the number of TPE evaluations (if early-stop never triggers)
MAX_EVALS = 1000

# (D) Stopping-criterion parameters:
#     If we see this many consecutive TPE trials each with relative improvement
#     < REL_IMPROV_THRESHOLD, we stop.
REL_IMPROV_THRESHOLD = 0.01  # 0.01 percent
MAX_CONSECUTIVE       = 10   # stop after 10 consecutive “tiny-improvement” trials

# ─────────────── Load DataCube & Build Lookup ───────────────

if not os.path.exists(DATA_CUBE_FILE):
    raise FileNotFoundError(f"DataCube not found at '{DATA_CUBE_FILE}'")

data_cube = pd.read_csv(DATA_CUBE_FILE, delimiter=";")
required_cols = {"Phosphate", "Temperature", "GrowthRate"}
if not required_cols.issubset(data_cube.columns):
    raise ValueError("DataCube.csv must contain columns: 'Phosphate', 'Temperature', and 'GrowthRate'")

data_cube["Phosphate"]   = data_cube["Phosphate"].round(0).astype(int)
data_cube["Temperature"] = data_cube["Temperature"].round(0).astype(int)

_lookup_dict = {
    (row.Phosphate, row.Temperature): row.GrowthRate
    for row in data_cube.itertuples(index=False)
}

def get_growth_rate(phosphate: int, temperature: int) -> float:
    """
    Return the GrowthRate for the integer (phosphate, temperature) pair.
    If no exact match, return np.nan.
    """
    return _lookup_dict.get((int(phosphate), int(temperature)), np.nan)

# ─────────────── Global State for Early-Stop Logic ───────────────

best_observed     = -np.inf   # best (maximum) growth seen so far
consecutive_small = 0         # how many consecutive “tiny/no improvements” we’ve seen

# ─────────────── Define the Search Space ───────────────

space = {
    "phosphate":   hp.quniform("phosphate", PHOSPHATE_BOUNDS[0], PHOSPHATE_BOUNDS[1], 1),
    "temperature": hp.quniform("temperature", TEMPERATURE_BOUNDS[0], TEMPERATURE_BOUNDS[1], 1),
}

# ─────────────── Define the Objective Function ───────────────

def objective(params):
    """
    Hyperopt objective: receives floats from hp.quniform, rounds them to ints,
    looks up true growth in DataCube, returns loss = -growth.

    Also updates global counters `best_observed` and `consecutive_small` to track
    whether we should stop early.
    """
    global best_observed, consecutive_small

    # 1. Round to nearest integer
    ph = int(round(params["phosphate"]))
    temp = int(round(params["temperature"]))

    # 2. Look up true growth
    growth = get_growth_rate(ph, temp)

    # 3. If not in DataCube, assign a large loss so TPE avoids it
    if np.isnan(growth):
        loss = 1e6
        observed_improvement = -np.inf
    else:
        loss = -growth
        # Compute relative improvement over best_observed so far
        if best_observed == -np.inf:
            # This is the first valid point seen
            observed_improvement = np.inf
        else:
            observed_improvement = (growth - best_observed) / max(1.0, best_observed) * 100.0

    # 4. Update stopping-criterion counters
    if (not np.isnan(growth)) and (growth > best_observed):
        # If this trial is strictly better than best_observed
        if observed_improvement < REL_IMPROV_THRESHOLD:
            consecutive_small += 1
        else:
            consecutive_small = 0
        best_observed = growth
    else:
        # No improvement (either NaN or ≤ best_observed)
        consecutive_small += 1

    # 5. Return Hyperopt result dictionary
    return {
        "loss": loss,
        "status": STATUS_OK,
        "phosphate": ph,
        "temperature": temp,
        "true_growth": float(growth) if not np.isnan(growth) else np.nan
    }

# ─────────────── Define the Early-Stop Function ───────────────

def early_stop_fn(trials, *args):
    """
    Called by Hyperopt after each trial. If it returns (True, {}), Hyperopt stops.

    We iterate over all completed trials (in order) and re-compute a local best_so_far.
    If we see MAX_CONSECUTIVE trials in a row whose improvement < REL_IMPROV_THRESHOLD
    (or no improvement), we return True to stop.
    """
    best_so_far      = -np.inf
    consecutive_count = 0

    for t in trials.trials:
        res = t["result"]
        g   = res.get("true_growth", np.nan)

        if np.isnan(g):
            # Invalid or missing → no improvement
            consecutive_count += 1
        else:
            if best_so_far == -np.inf:
                # First valid trial → consider infinite improvement
                improvement = np.inf
            else:
                improvement = (g - best_so_far) / max(1.0, best_so_far) * 100.0

            if g > best_so_far:
                # Strictly better than anything before
                if improvement < REL_IMPROV_THRESHOLD:
                    consecutive_count += 1
                else:
                    consecutive_count = 0
                best_so_far = g
            else:
                # g <= best_so_far, so no improvement
                consecutive_count += 1

        # Check if we have reached the threshold of consecutive small/no improvements
        if consecutive_count >= MAX_CONSECUTIVE:
            return True, {}

    # Otherwise, keep going
    return False, {}

# ─────────────── Run Hyperopt with Early-Stop Heuristic ───────────────

# Create a Trials() object to record all evaluations
trials = Trials()

# Use a time-based seed so that each run can be different
seed = int(time.time())

best = fmin(
    fn            = objective,
    space         = space,
    algo          = tpe.suggest,
    max_evals     = MAX_EVALS,
    trials        = trials,
    rstate        = np.random.default_rng(seed),
    early_stop_fn = early_stop_fn,
    show_progressbar = True
)

# 1. How many trials actually ran?
num_trials_done = len(trials.trials)

# 2. Extract final best parameters (floats → round to int)
best_ph   = int(round(best["phosphate"]))
best_temp = int(round(best["temperature"]))

# 3. Look up the final best growth (in case it was NaN)
best_growth = get_growth_rate(best_ph, best_temp)

# ─────────────── Display Results ───────────────

print("\n=================== Results ===================")
print(" TPE (hyperopt) with Early-Stop Heuristic")
print(f"  • Best Phosphate    = {best_ph}")
print(f"  • Best Temperature  = {best_temp}")
print(f"  • Best GrowthRate   = {best_growth:.6f}")
print(f"  • Total trials run  = {num_trials_done}")
if num_trials_done < MAX_EVALS:
    print(f"  • Stopped early after {num_trials_done} trials (< {MAX_EVALS})")
    print(f"    because ≥ {MAX_CONSECUTIVE} consecutive trials each had < {REL_IMPROV_THRESHOLD}% improvement.")
else:
    print(f"  • Reached max_evals = {MAX_EVALS}")
print("================================================\n")

# ─────────────── (Optional) Top 5 Points by True Growth ───────────────

all_results = []
for t in trials.trials:
    res = t["result"]
    g   = res.get("true_growth", np.nan)
    p   = res.get("phosphate", None)
    tmp = res.get("temperature", None)
    if not np.isnan(g):
        all_results.append((g, p, tmp))

all_results.sort(key=lambda x: x[0], reverse=True)

print("Top 5 evaluated points (by true growth):")
for i, (g, p, t) in enumerate(all_results[:5], start=1):
    print(f"  #{i}: Growth={g:.6f} at (Phosphate={p}, Temperature={t})")
print()

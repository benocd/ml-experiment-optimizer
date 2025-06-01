import numpy as np
import pandas as pd
import os
from scipy.stats.qmc import LatinHypercube
from sklearn.cluster import KMeans
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.spatial.distance import cdist

############################################################
# Configuration
############################################################

# Integer domain bounds
phosphate_bounds = (1, 20)     # integer
temperature_bounds = (4, 36)   # integer
max_loops = 50                 # safety limit

# Path to your DataCube CSV
data_cube_file = r"../DataCube.csv"

# We only want to test Bayesian Optimization with:
#   • batch size = 5
#   • EI‐threshold = 0.01%
EI_threshold_percent = 0.01
batch_size = 5

############################################################
# Load the data cube (rounded to 0 decimals)
############################################################

if not os.path.exists(data_cube_file):
    raise FileNotFoundError(f"DataCube not found at {data_cube_file}")

data_cube = pd.read_csv(data_cube_file, delimiter=";")
required_cols = {"Phosphate", "Temperature", "GrowthRate"}
if not required_cols.issubset(data_cube.columns):
    raise ValueError("DataCube.csv must contain columns: 'Phosphate', 'Temperature', and 'GrowthRate'")

# Round Phosphate/Temperature to integer
data_cube["Phosphate"] = data_cube["Phosphate"].round(0)
data_cube["Temperature"] = data_cube["Temperature"].round(0)


def get_growth_rate_from_datacube(phosphate, temperature):
    """
    Look up the growth rate from the data cube for integer phosphate & temperature.
    Returns np.nan if no exact match.
    """
    phos_i = int(round(phosphate, 0))
    temp_i = int(round(temperature, 0))
    subset = data_cube[
        (data_cube["Phosphate"] == phos_i) &
        (data_cube["Temperature"] == temp_i)
    ]
    if len(subset) == 1:
        return float(subset["GrowthRate"].values[0])
    else:
        return np.nan


############################################################
# Core Functions
############################################################

def fit_gp_model(df_):
    """
    Fit a Gaussian Process (with a Matern kernel) on the observed points.
    """
    X_ = df_[["Phosphate", "Temperature"]].values
    y_ = df_["GrowthRate"].values
    gp_ = GaussianProcessRegressor(
        kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True
    )
    gp_.fit(X_, y_)
    return gp_

def expected_improvement(X_candidates, gp_, y_best, xi=0.01):
    """
    Compute Expected Improvement (EI) for each candidate in X_candidates,
    given the current GP and the best observed y_best. 
    xi is the exploration‐exploitation tradeoff (as a fraction, not %).
    
    Returns an array of EI values (same length as X_candidates).
    """
    mu, sigma = gp_.predict(X_candidates, return_std=True)
    sigma = sigma.flatten()
    improvement = mu - y_best - xi
    # Avoid division by zero:
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0] = 0.0
    return ei

def run_bayesian_optimization(EI_threshold_percent, batch_size):
    """
    Pure Bayesian‐Optimization‐style loop (batch mode). Steps:

      1. Draw an initial LHS sample of size `batch_size`. Round to integers, look
         up GrowthRate from DataCube, drop any misses.
      2. Repeat (up to max_loops):
         a. Fit a GP to the current observed data.
         b. Compute EI on the full integer grid.
         c. Take top‐EI 300 grid points (or max available) → cluster them into `batch_size`.
         d. Compute EI for those `batch_size` points, scale each by max(current_best, 1.0).
         e. If all scaled EI (%) < EI_threshold_percent → STOP and return best‐observed.
         f. Otherwise, “measure” (lookup) the true growth at each of the `batch_size` points,
            append to the dataset, and continue.
      3. If max_loops is reached without meeting the threshold → return best‐observed so far.

    Returns:
      (loop_count, best_observed_growth, best_phosphate, best_temperature)
    """
    # 1. INITIAL LHS SAMPLE
    sampler = LatinHypercube(d=2)
    lhs_pts = sampler.random(batch_size)
    phos_scaled = phosphate_bounds[0] + lhs_pts[:, 0] * (phosphate_bounds[1] - phosphate_bounds[0])
    temp_scaled = temperature_bounds[0] + lhs_pts[:, 1] * (temperature_bounds[1] - temperature_bounds[0])

    df_obs = pd.DataFrame({
        "Phosphate": np.round(phos_scaled, 0),
        "Temperature": np.round(temp_scaled, 0),
    })
    # Look up GrowthRate in the data cube
    df_obs["GrowthRate"] = df_obs.apply(
        lambda row: get_growth_rate_from_datacube(row["Phosphate"], row["Temperature"]),
        axis=1
    )
    df_obs = df_obs.dropna(subset=["GrowthRate"]).reset_index(drop=True)

    # If no valid points came back, we cannot proceed
    if df_obs.empty:
        return (0, np.nan, np.nan, np.nan)

    loop_count = 0
    scale_threshold = EI_threshold_percent  # in %

    while loop_count < max_loops:
        loop_count += 1

        # (a) Fit GP to current data
        gp_model = fit_gp_model(df_obs)
        y_best_obs = df_obs["GrowthRate"].max()

        # (b) Compute EI on the full integer grid
        p_lin = np.arange(phosphate_bounds[0], phosphate_bounds[1] + 1)
        t_lin = np.arange(temperature_bounds[0], temperature_bounds[1] + 1)
        P_grid, T_grid = np.meshgrid(p_lin, t_lin)
        grid_points = np.vstack([P_grid.ravel(), T_grid.ravel()]).T

        ei_vals = expected_improvement(grid_points, gp_model, y_best_obs)

        # (c) Take top‐300 EI candidates (or fewer if the grid is smaller)
        top_count = min(len(ei_vals), 300)
        top_indices = np.argsort(ei_vals)[-top_count:]
        top_candidates = grid_points[top_indices]  # shape = (top_count, 2)

        # Cluster those top candidates down to `batch_size` using KMeans
        kmeans = KMeans(n_clusters=batch_size, n_init=10, random_state=42)
        kmeans.fit(top_candidates)
        # Find the actual grid points closest to each cluster center:
        cluster_indices = np.argmin(cdist(kmeans.cluster_centers_, top_candidates), axis=1)
        next_batch = top_candidates[cluster_indices]  # shape = (batch_size, 2)

        # (d) Compute EI for these next points and scale to percent of y_best_obs
        scale_factor = max(y_best_obs, 1.0)
        ei_next = expected_improvement(next_batch, gp_model, y_best_obs)
        ei_next_pct = (ei_next / scale_factor) * 100.0

        # (e) If all EI_next_pct < threshold → STOP
        if np.all(ei_next_pct < scale_threshold):
            # Identify which point gave y_best_obs and return its coords
            best_row = df_obs.loc[df_obs["GrowthRate"].idxmax()]
            return (
                loop_count,
                float(best_row["GrowthRate"]),
                int(best_row["Phosphate"]),
                int(best_row["Temperature"])
            )

        # (f) Otherwise, “measure” (lookup) growth for each point in next_batch
        new_data = []
        for phos, temp in next_batch:
            ph_i = int(round(phos, 0))
            t_i = int(round(temp, 0))
            growth = get_growth_rate_from_datacube(ph_i, t_i)
            new_data.append([ph_i, t_i, growth])

        new_df = pd.DataFrame(new_data, columns=["Phosphate", "Temperature", "GrowthRate"])
        new_df = new_df.dropna(subset=["GrowthRate"])
        # Append to observed set
        df_obs = pd.concat([df_obs, new_df], ignore_index=True)

    # If we exit because max_loops was reached
    best_row = df_obs.loc[df_obs["GrowthRate"].idxmax()]
    return (
        loop_count,
        float(best_row["GrowthRate"]),
        int(best_row["Phosphate"]),
        int(best_row["Temperature"])
    )

############################################################
# Main: run the Bayesian Optimization procedure once
############################################################

if __name__ == "__main__":
    loops, best_growth, best_phos, best_temp = run_bayesian_optimization(
        EI_threshold_percent=EI_threshold_percent,
        batch_size=batch_size
    )

    print("EI_threshold%, batch_size, loops, best_growth, best_phosphate, best_temperature")
    print(f"{EI_threshold_percent}, {batch_size}, {loops}, {best_growth:.4f}, {best_phos}, {best_temp}")

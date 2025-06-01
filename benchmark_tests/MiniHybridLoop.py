import numpy as np
import pandas as pd
import os
from scipy.stats.qmc import LatinHypercube
from sklearn.cluster import KMeans
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import statsmodels.formula.api as smf
from scipy.optimize import minimize
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

# We only want to test EI threshold = 0.01 % and sample size = 5
EI_threshold_percent = 0.01
num_points = 5

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

def fit_ols_model(df_):
    """
    Fit a quadratic OLS on Phosphate and Temperature.
    """
    df_ = df_.copy()
    df_["Phosphate_sq"] = df_["Phosphate"] ** 2
    df_["Temp_sq"] = df_["Temperature"] ** 2
    df_["Phosphate_Temp"] = df_["Phosphate"] * df_["Temperature"]
    model = smf.ols(
        "GrowthRate ~ Phosphate + Temperature + Phosphate_sq + Temp_sq + Phosphate_Temp",
        data=df_
    ).fit()
    return model

def ols_prediction(phosphate, temperature, params):
    return (
        params["Intercept"]
        + params["Phosphate"] * phosphate
        + params["Temperature"] * temperature
        + params["Phosphate_sq"] * phosphate**2
        + params["Temp_sq"] * temperature**2
        + params["Phosphate_Temp"] * phosphate * temperature
    )

def find_ols_optimum(ols_model):
    """
    Numerically find the OLS‐predicted maximum within the integer domain.
    """
    params = ols_model.params

    def objective(x):
        # We negate because we want to maximize
        return -ols_prediction(x[0], x[1], params)

    # Start from the midpoint of the domain
    x0 = [
        (phosphate_bounds[0] + phosphate_bounds[1]) // 2,
        (temperature_bounds[0] + temperature_bounds[1]) // 2
    ]
    bnds = [phosphate_bounds, temperature_bounds]
    res = minimize(objective, x0, bounds=bnds)
    best_phos, best_temp = res.x
    best_growth = -res.fun
    return best_phos, best_temp, best_growth

def fit_gp_model(df_):
    """
    Fit a Gaussian Process with a Matern kernel on the observed points.
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
    Compute EI for a set of candidates, given the current GP and best observed y.
    xi is the exploration‐exploitation tradeoff (as a fraction, not %).
    """
    mu, sigma = gp_.predict(X_candidates, return_std=True)
    sigma = sigma.flatten()
    improvement = mu - y_best - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    # If sigma is zero, EI = 0
    ei[sigma == 0] = 0.0
    return ei

def run_procedure(EI_threshold_percent, num_points):
    """
    Iterative procedure:
      1. Start with num_points points from a Latin Hypercube sample (rounded to integer).
      2. Fit OLS → find OLS optimum.
      3. Fit GP → compute EI on integer grid.
      4. Pick top‐300 EI candidates, cluster to num_points, evaluate their EI.
      5. If all EI (%) < threshold → STOP; else add those points & repeat.
    Returns: (loop_count, final_ols_growth, final_ols_phosphate, final_ols_temperature)
    """
    # 1. Initial Latin Hypercube sample
    sampler = LatinHypercube(d=2)
    lhs_sample = sampler.random(num_points)
    phos_scaled = phosphate_bounds[0] + lhs_sample[:, 0] * (phosphate_bounds[1] - phosphate_bounds[0])
    temp_scaled = temperature_bounds[0] + lhs_sample[:, 1] * (temperature_bounds[1] - temperature_bounds[0])

    df_ = pd.DataFrame({
        "Phosphate": np.round(phos_scaled, 0),
        "Temperature": np.round(temp_scaled, 0),
    })
    # Look up actual GrowthRate from data cube
    df_["GrowthRate"] = df_.apply(
        lambda row: get_growth_rate_from_datacube(row["Phosphate"], row["Temperature"]),
        axis=1
    )
    df_ = df_.dropna(subset=["GrowthRate"]).reset_index(drop=True)
    if df_.empty:
        return (0, np.nan, np.nan, np.nan)

    loop_count = 0
    scale_threshold = EI_threshold_percent  # in percent
    max_loops_local = max_loops

    while loop_count < max_loops_local:
        loop_count += 1

        # (a) Fit OLS & find its predicted optimum
        ols_model = fit_ols_model(df_)
        best_phos_ols, best_temp_ols, best_growth_ols = find_ols_optimum(ols_model)

        # (b) Fit GP to the observed points
        gp_model = fit_gp_model(df_)
        y_best_obs = df_["GrowthRate"].max()

        # (c) Compute EI over the integer grid
        p_lin = np.arange(phosphate_bounds[0], phosphate_bounds[1] + 1)
        t_lin = np.arange(temperature_bounds[0], temperature_bounds[1] + 1)
        P_grid, T_grid = np.meshgrid(p_lin, t_lin)
        grid_points = np.vstack([P_grid.ravel(), T_grid.ravel()]).T

        ei_vals = expected_improvement(grid_points, gp_model, y_best_obs)

        # (d) Take top‐300 EI points (or fewer, if grid is smaller)
        top_count = min(len(ei_vals), 300)
        top_indices = np.argsort(ei_vals)[-top_count:]
        top_candidates = grid_points[top_indices]

        # Cluster those top candidates down to num_points
        kmeans = KMeans(n_clusters=num_points, n_init=10, random_state=42)
        kmeans.fit(top_candidates)
        cluster_indices = np.argmin(cdist(kmeans.cluster_centers_, top_candidates), axis=1)
        next_conditions = top_candidates[cluster_indices]

        # (e) Compute EI (%) at those new points, scaling by max(best_growth_ols, 1.0)
        scale_factor = max(best_growth_ols, 1.0)
        ei_for_next = expected_improvement(next_conditions, gp_model, y_best_obs)
        ei_for_next_pct = (ei_for_next / scale_factor) * 100.0

        # (f) STOP if all new EI (%) < threshold
        if np.all(ei_for_next_pct < scale_threshold):
            return (loop_count, best_growth_ols, best_phos_ols, best_temp_ols)

        # (g) Otherwise, “measure” growth at those points and add to df_
        new_data = []
        for phos, temp in next_conditions:
            ph_i = int(round(phos, 0))
            t_i = int(round(temp, 0))
            growth = get_growth_rate_from_datacube(ph_i, t_i)
            new_data.append([ph_i, t_i, growth])

        new_df = pd.DataFrame(new_data, columns=["Phosphate", "Temperature", "GrowthRate"])
        new_df = new_df.dropna(subset=["GrowthRate"])
        df_ = pd.concat([df_, new_df], ignore_index=True)

    # If we hit max_loops without meeting the threshold
    return (loop_count, best_growth_ols, best_phos_ols, best_temp_ols)


############################################################
# Main: run only for EI_threshold=0.01% and num_points=5
############################################################

if __name__ == "__main__":
    loops, final_growth, final_phos, final_temp = run_procedure(
        EI_threshold_percent=EI_threshold_percent,
        num_points=num_points
    )
    print("EI_threshold%, num_points, loops, final_OLS_growth, final_phosphate, final_temperature")
    print(f"{EI_threshold_percent}, {num_points}, {loops}, {final_growth:.4f}, {final_phos:.2f}, {final_temp:.2f}")

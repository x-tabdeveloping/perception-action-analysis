"""Script for analyzing mouse tracking data with the bezier-trajectory model.
The script is desinged for running in an IPython based interactive shell.
"""
# %% Loading Ipython autoreload
from IPython import get_ipython

ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

# %% Loading all else
import arviz as az
import pymc as pm
import pandas as pd
import numpy as np

from utils.plots import visualize_mouse_data

# %% Loading experiment data
dat = pd.read_csv("../dat/mouse_data.csv")
dat["participant_id"], participant_coords = pd.factorize(dat.doc_id)

# %% Plot for mouse tracking data, so I can look at all of it at once
fig = visualize_mouse_data(dat, x="mouse_x", y="mouse_y")
fig.show()

# %% Model definition
coords = {
    "condition_levels": ["control", "experimental"],
    "nationality_levels": ["danish", "hungarian"],
    "participant": participant_coords,
    "trial": np.arange(20),
}

with pm.Model(coords=coords) as model:
    # ---Declaring data---
    d = pm.MutableData("distance", dat.cum_dist.fillna(0))
    condition = pm.MutableData("condition", dat.condition)
    nationality = pm.MutableData("nationality", dat.nationality)
    participant_id = pm.MutableData("participant_id", dat.participant_id)
    i_trial = pm.MutableData("i_trial", dat.i_trial)

    # Population level
    pop_mu_t = pm.Normal("pop_mu_t", 0.5, 1.0)
    pop_sd_t = pm.HalfCauchy("pop_sd_t", 1.0)
    pop_mu_s = pm.Normal("pop_mu_s", 0.0, 1.0)
    pop_sd_s = pm.HalfCauchy("pop_sd_s", 1.0)
    pop_effect_mu_t = pm.Normal("pop_effect_mu_t", 0.0, 1.0)
    pop_effect_sd_t = pm.HalfCauchy("pop_effect_sd_t", 1.0)
    pop_effect_mu_s = pm.Normal("pop_effect_mu_s", 0.0, 1.0)
    pop_effect_sd_s = pm.HalfCauchy("pop_effect_sd_s", 1.0)

    # Nation level
    nation_mu_t = pm.Normal(
        "nation_mu_t", pop_mu_t, pop_sd_t, dims=("nationality_levels")
    )
    nation_sd_t = pm.HalfCauchy(
        "nation_sd_t", 1.0, dims=("nationality_levels")
    )
    nation_mu_s = pm.Normal(
        "nation_mu_s", pop_mu_s, pop_sd_s, dims=("nationality_levels")
    )
    nation_sd_s = pm.HalfCauchy(
        "nation_sd_s", 1.0, dims=("nationality_levels")
    )
    nation_effect_mu_t = pm.Normal(
        "nation_effect_mu_t",
        pop_effect_mu_t,
        pop_effect_sd_t,
        dims=("nationality_levels"),
    )
    nation_effect_sd_t = pm.HalfCauchy(
        "nation_effect_sd_t", 1.0, dims=("nationality_levels")
    )
    nation_effect_mu_s = pm.Normal(
        "nation_effect_mu_s",
        pop_effect_mu_s,
        pop_effect_sd_s,
        dims=("nationality_levels"),
    )
    nation_effect_sd_s = pm.HalfCauchy(
        "nation_effect_sd_s", 1.0, dims=("nationality_levels")
    )
    error_t = pm.HalfCauchy("error_t", 1.0)
    error_s = pm.HalfCauchy("error_s", 1.0)

    # Personal level
    personal_mu_t = pm.Normal(
        "personal_mu_t",
        nation_mu_t,
        nation_sd_t,
        dims=("participant", "nationality_levels"),
    )
    personal_sd_t = pm.HalfCauchy(
        "personal_sd_t", 1.0, dims=("participant", "nationality_levels")
    )
    personal_mu_s = pm.Normal(
        "personal_mu_s",
        nation_mu_s,
        nation_sd_s,
        dims=("participant", "nationality_levels"),
    )
    personal_sd_s = pm.HalfCauchy(
        "personal_sd_s", 1.0, dims=("participant", "nationality_levels")
    )
    personal_effect_t = pm.Normal(
        "personal_effect_t",
        nation_effect_mu_t,
        nation_effect_sd_t,
        dims=("participant", "nationality_levels"),
    )
    personal_effect_s = pm.Normal(
        "personal_effect_s",
        nation_effect_mu_s,
        nation_effect_sd_s,
        dims=("participant", "nationality_levels"),
    )

    # Trial level
    t = pm.Normal(
        "t",
        personal_mu_t,
        personal_sd_t,
        dims=("trial", "participant", "nationality_levels"),
    )
    s = pm.Normal(
        "s",
        personal_mu_s,
        personal_sd_s,
        dims=("trial", "participant", "nationality_levels"),
    )

    # ---Bezier outcome---
    # Calculating control point
    t = (
        t[i_trial, participant_id, nationality]
        + personal_effect_t[participant_id, nationality] * condition
    )
    s = (
        s[i_trial, participant_id, nationality]
        + personal_effect_s[participant_id, nationality] * condition
    )
    # Means for data points
    mu_x = -2 * pm.math.sqr(d) * t + 2 * d * t + pm.math.sqr(d)
    mu_y = -2 * pm.math.sqr(d) * s + 2 * d * s
    # Outcome distribution
    x = pm.Normal("x", mu_x, error_t, observed=dat.mouse_x)
    y = pm.Normal("y", mu_y, error_s, observed=dat.mouse_y)

# %% Sampling model with ADVI
with model:
    trace = pm.sample_prior_predictive()
    # Variational inference
    variational_fit = pm.fit(n=50000, method="advi")
    # Obtaining posterior sample
    trace.extend(variational_fit.sample())
    trace.extend(pm.sample_posterior_predictive(trace))

# %% Plotting nation level effect sizes
az.plot_forest(trace, var_names={"nation_effect_mu"}, filter_vars="like")

# %% Plotting interpersonal variabilities in effects
az.plot_forest(trace, var_names={"nation_effect_sigma"}, filter_vars="like")

# %% Creating summary of population and nation level parameters
summary = az.summary(trace, var_names={"pop", "nation"}, filter_vars="like")
# Saving summary as CSV
summary.to_csv("../dat/summary.csv")

# %% Plotting trace
az.plot_trace(trace, compact=True)

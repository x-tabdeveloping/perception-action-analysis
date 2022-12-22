# %% Loading Ipython autoreload
from IPython import get_ipython

ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")
# %% Loading all else
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.plots import visualize_mouse_data

# %% Loading experiment data
dat = pd.read_csv("../dat/mouse_data.csv")
dat["participant_id"], participant_coords = pd.factorize(dat.doc_id)
# %% Plot for mouse tracking data, so I can look at all of it at once
fig = visualize_mouse_data(dat, x="mouse_x", y="mouse_y")
fig.show()

# %%
mouse = dat[(dat.doc_id == "yU0N886Y4PfoGF6LiZWT") & (dat.i_trial == 5)]
mouse

# %%
def bezier(time: float, strength: float):
    p0 = np.array((0, 0))
    p2 = np.array((1, 0))
    p1 = np.array((time, strength))

    def _bezier(t):
        t = np.array((t, t)).T
        return (1 - t) * ((1 - t) * p0 + t * p1) + t * ((1 - t) * p1 + t * p2)

    return _bezier


# %%
curve = bezier(0.61, 0.5)
mouse["bezier_x"], mouse["bezier_y"] = curve(mouse.cum_dist).T
fig = px.scatter(mouse, x="mouse_x", y="mouse_y")
fig.add_trace(go.Scatter(x=mouse.bezier_x, y=mouse.bezier_y, mode="markers"))
fig.show()

# %%
dat.doc_id + dat.i_trial.astype(str)

# %%
dat.doc_id.nunique()

# %% Model definition
import pymc as pm


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

    # ---Intercept parameters---

    # Population level
    pop_mu_time = pm.Normal("pop_mu_time", 0.5, 0.5)
    pop_sd_time = pm.HalfCauchy("pop_sd_time", 0.1)
    pop_mu_strength = pm.Normal("pop_mu_strength", 0.0, 0.5)
    pop_sd_strength = pm.HalfCauchy("pop_sd_strength", 0.1)
    error = pm.HalfCauchy("error", 0.01)

    # Personal level
    personal_mu_time = pm.Normal(
        "personal_mu_time", pop_mu_time, pop_sd_time, dims=("participant")
    )
    personal_sd_time = pm.HalfCauchy(
        "personal_sd_time", 0.1, dims=("participant")
    )
    personal_mu_strength = pm.Normal(
        "personal_mu_strength",
        pop_mu_strength,
        pop_sd_strength,
        dims=("participant"),
    )
    personal_sd_strength = pm.HalfCauchy(
        "personal_sd_strength", 0.1, dims=("participant")
    )

    # Trial level
    time = pm.Normal(
        "time",
        pop_mu_time,
        pop_sd_time,
        dims=("trial"),
    )
    strength = pm.Normal(
        "strength",
        pop_mu_strength,
        pop_sd_strength,
        dims=("trial"),
    )

    # ---Effect parameters---
    # Population level
    mu_effect_t = pm.Normal("mu_effect_time", 0.0, 0.1)
    mu_effect_s = pm.Normal("mu_effect_strength", 0.0, 0.1)
    sd_effect_t = pm.HalfCauchy("sd_effect_time", 0.1)
    sd_effect_s = pm.HalfCauchy("sd_effect_strength", 0.1)
    # Nation level
    effect_t = pm.Normal(
        "nation_effect_time", 0.0, 0.1, dims=("nationality_levels")
    )
    effect_s = pm.Normal(
        "nation_effect_strength", 0.0, 0.1, dims=("nationality_levels")
    )

    # ---Bezier outcome---
    # Calculating control point
    t = time[i_trial] + effect_t[nationality] * condition
    s = strength[i_trial] + effect_s[nationality] * condition
    # Means for data points
    mu_x = -2 * pm.math.sqr(d) * t + 2 * d * t + pm.math.sqr(d)
    mu_y = -2 * pm.math.sqr(d) * s + 2 * d * s
    # Outcome distribution
    x = pm.Normal("x", mu_x, error, observed=dat.mouse_x)
    y = pm.Normal("y", mu_y, error, observed=dat.mouse_y)

# %%
with model:
    trace = pm.sample_prior_predictive()
    variational_fit = pm.fit(n=50000, method="advi")
    trace.extend(variational_fit.sample())
    trace.extend(pm.sample_posterior_predictive(trace))

# %%
import arviz as az

az.plot_forest(trace, var_names={"nation"}, filter_vars="like")

# %%
az.plot_dist_comparison(trace)


# %%
az.plot_ppc(trace, group="prior")

# %%
az.plot_ppc(trace)

# %%
az.plot_trace(trace)

# %%

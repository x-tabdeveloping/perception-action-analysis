import sys

import pandas as pd
import numpy as np
import pyddm as ddm
from pyddm.plot import model_gui

sys.setrecursionlimit(10000)

dat = pd.read_csv("../dat/rt_data.csv", index_col=0)
dat = dat.assign(
    condition=np.where(dat.condition == "control", 0, 1)
)
dat = dat.assign(
    nationality=np.where(dat.nationality == "danish", 0, 1)
)
sample = ddm.Sample.from_pandas_dataframe(
    dat, rt_column_name="rt", correct_column_name="response"
)

class DriftRegressor(ddm.models.Drift):
    name = (
        "Drift is linearly regressed over nationality,"
        "condition and their interaction"
    )
    required_parameters = [
        "intercept", 
        "beta_nationality",
        "beta_condition",
        "beta_interaction",
        "leak"
    ]
    required_conditions = ["nationality", "condition"]

    def get_drift(self, x, conditions, **kwargs):
        return (
            self.intercept
            + self.beta_nationality * conditions["nationality"]
            + self.beta_condition * conditions["condition"]
            + self.beta_interaction * conditions["nationality"] * conditions["condition"]
            + self.leak * x
        )

drift = DriftRegressor(
    intercept=ddm.Fittable(minval=0, maxval=10.0),
    beta_nationality=ddm.Fittable(minval=-2, maxval=2),
    beta_condition=ddm.Fittable(minval=-2, maxval=2),
    beta_interaction=ddm.Fittable(minval=-2, maxval=2),
    leak=ddm.Fittable(minval=-10, maxval=10),
)

model = ddm.Model(
    drift=drift,
    noise=ddm.NoiseConstant(noise=1),
    overlay=ddm.OverlayNonDecision(
        nondectime=ddm.Fittable(minval=0, maxval=1.0)
    ),
    bound=ddm.BoundCollapsingExponential(
        B=ddm.Fittable(minval=0.5, maxval=3),
        tau=ddm.Fittable(minval=.0001, maxval=5)
    ),
    T_dur=10.0,
    dx=0.01,
    dt=0.01
)
model = ddm.Model(drift=ddm.DriftConstant(drift=ddm.Fittable(minval=0, maxval=10.0)), T_dur=10.0)
fit = ddm.fit_adjust_model(sample, model)
model_gui(model, sample)




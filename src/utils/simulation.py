"""Utilities for simulating experiment data."""

import pandas as pd
import numpy as np


def simulate_data(
    n_participants: int, n_experimental: int, n_control: int
) -> pd.DataFrame:
    participants = pd.DataFrame(
        dict(
            doc_id=np.arange(n_participants * 2),
            nationality=["hungarian"] * n_participants
            + ["danish"] * n_participants,
            mean_reaction_time=np.random.normal(
                7, 0.1, size=n_participants * 2
            ),
            mean_auc=np.random.normal(-1, 0.2, size=n_participants * 2),
        )
    )
    participants["auc_effect"] = np.where(
        participants["nationality"] == "danish",
        np.random.normal(1.0, 0.5),
        np.random.normal(2.5, 0.1),
    )
    trials = participants.loc[
        participants.index.repeat(n_experimental + n_control)
    ]
    trials["condition"] = np.tile(
        ["experimental"] * n_experimental + ["control"] * n_control,
        n_participants * 2,
    )
    trials["chosen"] = np.where(
        trials.nationality == "hungarian",
        np.where(
            trials.condition == "experimental",
            np.where(
                np.random.binomial(n=1, p=0.8, size=len(trials.index)) == 1,
                "correct",
                "incorrect",
            ),
            np.where(
                np.random.binomial(n=1, p=0.95, size=len(trials.index)) == 1,
                "correct",
                "incorrect",
            ),
        ),
        np.where(
            trials.condition == "experimental",
            np.where(
                np.random.binomial(n=1, p=0.9, size=len(trials.index)) == 1,
                "correct",
                "incorrect",
            ),
            np.where(
                np.random.binomial(n=1, p=0.95, size=len(trials.index)) == 1,
                "correct",
                "incorrect",
            ),
        ),
    )
    trials["reaction_time"] = np.where(
        trials.nationality == "hungarian",
        np.where(
            trials.condition == "experimental",
            np.random.lognormal(trials.mean_reaction_time + 0.6, 0.2),
            np.random.lognormal(trials.mean_reaction_time, 0.2),
        ),
        np.where(
            trials.condition == "experimental",
            np.random.lognormal(trials.mean_reaction_time + 0.2, 0.2),
            np.random.lognormal(trials.mean_reaction_time, 0.2),
        ),
    )
    trials["auc"] = np.where(
        trials.condition == "experimental",
        np.random.normal(trials.mean_auc + trials.auc_effect, 0.5),
        np.random.normal(trials.mean_auc, 0.5),
    )
    return trials

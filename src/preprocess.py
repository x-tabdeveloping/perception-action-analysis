"""Script for fetching and preprocessing data from the database.
Has to run in an environment that has firebase-admin installed.
"""
import os

import numpy as np

from utils.mouse import (
    add_velocity,
    expand_mouse_tracking_data,
    normalize_mouse_tracking_data,
)

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/home/au689890/firebase_credentials.json"


def main() -> None:
    # I have to import at runtime, because I set the firestore credentials
    # before this
    from utils.fetch import fetch_experiment_data

    print("Fetching data")
    data = fetch_experiment_data()
    data.to_pickle("../dat/data.pkl")
    # Preparing data for mouse tracking analysis
    print("Preparing mouse tracking data...")
    mouse_data = data
    print(f"N trials: {len(mouse_data.index)}")
    mouse_data = mouse_data[mouse_data.chosen == "correct"]
    print(f"N correct: {len(mouse_data.index)}")
    mouse_data = normalize_mouse_tracking_data(data)
    mouse_data = add_velocity(mouse_data)
    mouse_data = expand_mouse_tracking_data(mouse_data)
    mouse_data = mouse_data.assign(
        condition=np.where(mouse_data.condition == "control", 0, 1)
    )
    mouse_data = mouse_data.assign(
        nationality=np.where(mouse_data.nationality == "danish", 0, 1)
    )
    print("Saving")
    mouse_data.to_csv("../dat/mouse_data.csv")
    # Preparing reaction time data for drift diffusion modelling
    print("Preparing reaction time data...")
    rt_data = data[
        [
            "doc_id",
            "i_trial",
            "sex",
            "nationality",
            "condition",
            "chosen",
            "reaction_time",
        ]
    ]
    rt_data = rt_data.assign(
        response=np.where(rt_data.chosen == "correct", 1.0, 0.0)
    )
    rt_data = rt_data.rename(
        columns={
            "doc_id": "subj_idx",
            "reaction_time": "rt",
        }
    )
    # Converting to seconds
    rt_data["rt"] = rt_data.rt / 1000.0
    print("Saving")
    rt_data.to_csv("../dat/rt_data.csv")
    print("DONE")


if __name__ == "__main__":
    main()

# %%
import os

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/home/au689890/firebase_credentials.json"

# %%
import pandas as pd
import numpy as np

from utils.fetch import fetch_experiment_data
from utils.mouse import auc

# %%
dat = fetch_experiment_data()

# %%
dat
# %%
mouse = dat.mouseTrackingData.loc[3]

# %%
auc(mouse)

# %%
dat["auc"] = dat.mouseTrackingData.map(auc) / (
    dat.screenWidth * dat.screenHeight
)
dat

# %%
mouse_df = pd.DataFrame.from_records(mouse)
# %%
mouse_df = pd.DataFrame.from_records(mouse)
mouse_df["dx"] = mouse_df["x"] - mouse_df["x"].shift()
mouse_df["dy"] = mouse_df["y"] - mouse_df["y"].shift()
mouse_df["dt"] = mouse_df["timestamp"] - mouse_df["timestamp"].shift()
mouse_df["distance"] = np.sqrt(mouse_df["dx"] ** 2 + mouse_df["dy"] ** 2)
mouse_df["velocity"] = mouse_df["distance"] / mouse_df["dt"]
mouse_df

# %%

# %%
import os

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/home/au689890/firebase_credentials.json"

# %%
import pandas as pd
import numpy as np

from utils.fetch import fetch_experiment_data
from utils.mouse import aggregate_mouse_tracking

# %%
dat = fetch_experiment_data()

# %%
dat
# %%
aggregate_mouse_tracking(dat)
# %%

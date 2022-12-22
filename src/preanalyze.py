# %%
import pandas as pd
import plotly.express as px

# %%
dat = pd.read_csv("../dat/rt_data.csv")

# %%
px.histogram(dat.groupby("subj_idx").first(), x="nationality")

# %%

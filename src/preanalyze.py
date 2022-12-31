"""Script for checking distribution of nationalities and sexes."""
# %%
import pandas as pd
import plotly.express as px

# %%
dat = pd.read_csv("../dat/rt_data.csv")

# %%
participants = dat.groupby("subj_idx").first()

# %%
px.histogram(participants, x="nationality").show()

# %%
px.histogram(participants, x="sex")

# %%
px.box(dat, x="rt", color="nationality", y="condition")

# %%

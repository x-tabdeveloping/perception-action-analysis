import plotly.graph_objects as go
import pandas as pd


def visualize_mouse_data(
    data: pd.DataFrame, x: str = "mouse_x", y: str = "mouse_y"
) -> go.Figure:
    """Visualizes all mouse tracking data for all trials with a slider."""
    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for name, mouse_data in data.groupby(["doc_id", "i_trial"]):
        fig.add_trace(
            go.Scatter(
                visible=False,
                name=str(name),
                x=mouse_data[x],
                y=mouse_data[y],
                marker=dict(color=mouse_data.velocity),
                mode="markers",
            )
        )
    fig.data[0].visible = True
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": "Slider switched to step: " + str(i)},
            ],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    sliders = [
        dict(
            active=10,
            currentvalue={"prefix": "Mouse data:"},
            pad={"t": 50},
            steps=steps,
        )
    ]
    fig.update_layout(sliders=sliders, yaxis_range=[-0.2, 1])
    return fig

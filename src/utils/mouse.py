"""Utilities for manipulating mouse tracking data"""
from typing import TypedDict, List, Tuple

import numpy as np
import pandas as pd
from shapely import Polygon


class MouseDatapoint(TypedDict):
    y: int
    x: int
    timestamp: int


def area_shoelace(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate area of polygon with the shoelace formula."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def auc(tracking_data: List[MouseDatapoint]) -> float:
    """Calculates area under the curve values for mouse tracking data.

    Parameters
    ----------
    tracking_data: list of mouse tracking datapoints
        mouse tracking data for a single trial.

    Returns
    -------
    float
        Area under the curve.
    """
    tracking_df = pd.DataFrame.from_records(tracking_data)
    x = np.array(tracking_df.x)
    y = np.array(tracking_df.y)
    polygon = Polygon(zip(x, y))
    return polygon.area


def max_velocity(tracking_data: List[MouseDatapoint]) -> Tuple[float, int]:
    """Calculates maximum velocity for mouse tracking data.

    Parameters
    ----------
    tracking_data: list of mouse tracking datapoints
        mouse tracking data for a single trial.

    Returns
    -------
    max_velocity: float
        Maximum normalized velocity of the mouse.
    max_velocity_time: float
        Time of maximum velocity (in ms).
    """
    tracking_df = pd.DataFrame.from_records(tracking_data)
    x = tracking_df.x
    y = tracking_df.y
    dx = x - x.shift()
    dy = y - y.shift()
    dt = tracking_df.timestamp - tracking_df.timestamp.shift()
    dist = np.sqrt(dx**2 + dy**2)
    velocity = dist / dt
    max_v_i = np.argmax(velocity)
    max_velocity = velocity[max_v_i]
    max_velocity_time = tracking_df.timestamp.iloc[max_v_i]
    return max_velocity, max_velocity_time


def aggregate_mouse_tracking(experiment_data: pd.DataFrame) -> pd.DataFrame:
    """Calculates AUC, maximum velocity and maximum velocity times for mouse
    tracking data for each trial in the experiment.

    Parameters
    ----------
    experiment_data: DataFrame
        DataFrame containing trial data from the experiment.

    Returns
    -------
    DataFrame
        Extended data frame with mouse aggregated mouse tracking data.
    """
    experiment_data = experiment_data.copy()
    experiment_data[
        ["max_velocity", "max_velocity_time"]
    ] = experiment_data.mouse_tracking_data.map(max_velocity).tolist()
    experiment_data["auc"] = experiment_data.mouse_tracking_data.map(auc)
    return experiment_data

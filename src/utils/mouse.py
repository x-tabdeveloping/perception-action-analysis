"""Utilities for manipulating mouse tracking data"""
from typing import TypedDict, List, Tuple

import numpy as np
import pandas as pd


class MouseDatapoint(TypedDict):
    y: int
    x: int
    timestamp: int


def area_shoelace(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate area of polygon with the shoelace formula."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def auc(
    tracking_data: List[MouseDatapoint], screen_width: int, screen_height: int
) -> float:
    """Calculates area under the curve values for mouse tracking data.

    Parameters
    ----------
    tracking_data: list of mouse tracking datapoints
        mouse tracking data for a single trial.
    screen_width: int
        width of the participant's screen.
    screen_height: int
        height of the participant's screen.

    Returns
    -------
    float
        Area under the curve.
    """
    tracking_df = pd.DataFrame.from_records(tracking_data)
    x = np.array(tracking_df.x) / screen_width
    y = np.array(tracking_df.y) / screen_height
    return area_shoelace(x, y)


def max_velocity(
    tracking_data: List[MouseDatapoint], screen_width: int, screen_height: int
) -> Tuple[float, int]:
    """Calculates maximum velocity for mouse tracking data.

    Parameters
    ----------
    tracking_data: list of mouse tracking datapoints
        mouse tracking data for a single trial.
    screen_width: int
        width of the participant's screen.
    screen_height: int
        height of the participant's screen.

    Returns
    -------
    max_velocity: float
        Maximum normalized velocity of the mouse.
    max_velocity_time: float
        Time of maximum velocity (in ms).
    """
    tracking_df = pd.DataFrame.from_records(tracking_data)
    x = tracking_df.x / screen_width
    y = tracking_df.y / screen_height
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
    pass

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


def rot_matrix(theta: float):
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def min_max_norm(a):
    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    return a


def _normalize(
    tracking_data: List[MouseDatapoint],
    screen_height: int,
    reaction_time: float,
) -> List[MouseDatapoint]:
    tracking_df = pd.DataFrame.from_records(tracking_data)
    x, y = tracking_df.x, tracking_df.y
    x0, x1 = x.iloc[0], x.iloc[-1]
    y0, y1 = y.iloc[0], y.iloc[-1]
    dx = x1 - x0
    dy = y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    x = x / dist - 1
    if x.iloc[-1] < 0:
        x = -x
    y = 1 - (y / dist)
    x0, x1 = x.iloc[0], x.iloc[-1]
    y0, y1 = y.iloc[0], y.iloc[-1]
    x = x - x0
    y = y - y0
    slope = (y1 - y0) / (x1 - x0)
    rot = rot_matrix(-np.arctan(slope))
    x, y = np.dot(np.array((x, y)).T, rot.T).T
    timestamps = tracking_df.timestamp
    tracking_df["timestamp"] = timestamps / reaction_time
    tracking_df.x = x
    tracking_df.y = y
    return tracking_df.to_dict(orient="records")  # type: ignore


def normalize_mouse_tracking_data(
    experiment_data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalizes mouse tracking coordinates based on screen size."""
    return experiment_data.assign(
        mouse_tracking_data=experiment_data.apply(
            lambda row: _normalize(
                row.mouse_tracking_data, row.screen_height, row.reaction_time
            ),  # type: ignore
            axis=1,
        )
    )


def _add_velocity(tracking_data: List[MouseDatapoint]) -> List[MouseDatapoint]:
    tracking_df = pd.DataFrame.from_records(tracking_data)
    x = tracking_df.x
    y = tracking_df.y
    dx = x - x.shift()
    dy = y - y.shift()
    dt = tracking_df.timestamp - tracking_df.timestamp.shift()
    dist = np.sqrt(dx**2 + dy**2)
    velocity = dist / dt
    tracking_df["velocity"] = velocity
    tracking_df["distance"] = dist
    cum_dist = np.cumsum(dist)
    cum_dist = cum_dist / np.max(cum_dist)
    tracking_df["cum_dist"] = cum_dist
    return tracking_df.to_dict(orient="records")  # type: ignore


def add_velocity(experiment_data: pd.DataFrame) -> pd.DataFrame:
    """Adds velocities to mouse tracking data"""
    return experiment_data.assign(
        mouse_tracking_data=experiment_data.mouse_tracking_data.map(
            _add_velocity
        )
    )


def expand_mouse_tracking_data(experiment_data: pd.DataFrame) -> pd.DataFrame:
    """Expands mouse tracking data in the dataframe to different columns."""
    experiment_data = experiment_data.explode("mouse_tracking_data")
    experiment_data["mouse_x"] = experiment_data.mouse_tracking_data.map(
        lambda point: point["x"]
    )
    experiment_data["mouse_y"] = experiment_data.mouse_tracking_data.map(
        lambda point: point["y"]
    )
    experiment_data["timestamp"] = experiment_data.mouse_tracking_data.map(
        lambda point: point["timestamp"]
    )
    experiment_data["velocity"] = experiment_data.mouse_tracking_data.map(
        lambda point: point["velocity"]
    )
    experiment_data["distance"] = experiment_data.mouse_tracking_data.map(
        lambda point: point["distance"]
    )
    experiment_data["cum_dist"] = experiment_data.mouse_tracking_data.map(
        lambda point: point["cum_dist"]
    )
    experiment_data["cum_dist"] = experiment_data.cum_dist.fillna(0)
    experiment_data = experiment_data.drop(columns="mouse_tracking_data")
    return experiment_data

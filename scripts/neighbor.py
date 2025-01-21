import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from scipy.spatial import KDTree
from typing import Union

import utils

OPPOSITE_DIRECTIONS = {
    "Northbound": "Southbound",
    "Southbound": "Northbound",
    "Eastbound": "Westbound",
    "Westbound": "Eastbound",
    "Unknown": ""
}

def nearest_snap(line: Union[shapely.LineString, np.ndarray], point: shapely.Point, k_neighbors: int = 1) -> np.ndarray:
    """
    Based off of this function,
    but we want to return the index value, rather than the point.
    https://github.com/UTEL-UIUC/gtfs_segments/blob/main/gtfs_segments/geom_utils.py
    """
    if isinstance(line, shapely.LineString):
        line = np.asarray(line.coords)
    elif isinstance(line, np.ndarray):
        line = line
    point = np.asarray(point.coords)
    tree = KDTree(line)

    # np_dist is array of distances of result (let's not return it)
    # np_inds is array of indices of result
    _, np_inds = tree.query(
        point,
        workers=-1,
        k=k_neighbors,
    )

    return np_inds.squeeze()

def find_nearest_points(
    vp_coords_array: np.ndarray, 
    target_stop: shapely.Point, 
    vp_idx_array: np.ndarray,
) -> np.ndarray:
    """    
    vp_coords_line is all the vehicle positions strung together as 
    coordinates in a linestring.
    The target point is a stop.

    We want to find the k nearest points before/after a stop.
    Start with k=5.
    Returns an array that gives the indices that are the nearest k points 
    (ex: nearest 5 vp to each stop).
    """
    indices = nearest_snap(
        vp_coords_array, 
        target_stop, 
        k_neighbors = 5
    )
        
    # nearest neighbor returns self.N 
    # if there are no nearest neighbor results found
    # if we want 10 nearest neighbors and 8th, 9th, 10th are all
    # the same result, the 8th will have a result, then 9th and 10th will
    # return the length of the array (which is out-of-bounds)
    # using vp_coords_array keeps too many points (is this because coords can be dupes?)
    indices2 = indices[indices < vp_idx_array.size]
    
    return indices2


def filter_to_nearest2_vp(
    nearest_vp_coords_array: np.ndarray,
    shape_geometry: shapely.LineString,
    nearest_vp_idx_array: np.ndarray,
    stop_meters: float,
) -> tuple[np.ndarray]:
    """
    Take the indices that are the nearest.
    Filter the vp coords down and project those against the shape_geometry.
    Calculate how close those nearest k vp are to a stop (as they travel along a shape).
    
    Filter down to the nearest 2 vp before and after a stop.
    If there isn't one before or after, a value of -1 is returned.
    """
    # Project these vp coords to shape geometry and see how far it is
    # from the stop's position on the shape
    nearest_vp_projected = np.asarray(
        [shape_geometry.project(shapely.Point(i)) 
         for i in nearest_vp_coords_array]
    )

    # Negative values are before the stop
    # Positive values are vp after the stop
    before_indices = np.where(nearest_vp_projected - stop_meters < 0)[0]
    after_indices = np.where(nearest_vp_projected - stop_meters > 0)[0]
    
    # Set missing values when we're not able to find a nearest neighbor result
    # use -1 as vp_idx (since this is not present in vp_usable)
    # and zeroes for meters
    before_idx = -1
    after_idx = -1
    before_vp_meters = 0
    after_vp_meters = 0
    
    # Grab the closest vp before a stop (-1 means array was empty)
    if before_indices.size > 0:
        before_idx = nearest_vp_idx_array[before_indices][-1] 
        before_vp_meters = nearest_vp_projected[before_indices][-1]
   
    # Grab the closest vp after a stop (-1 means array was empty)
    if after_indices.size > 0:
        after_idx = nearest_vp_idx_array[after_indices][0]
        after_vp_meters = nearest_vp_projected[after_indices][0]
    
    return before_idx, after_idx, before_vp_meters, after_vp_meters


def two_nearest_neighbor_near_stop(
    vp_direction_array: np.ndarray,
    vp_geometry: shapely.LineString,
    vp_idx_array: np.ndarray,
    stop_geometry: shapely.Point,
    opposite_stop_direction: str,
    shape_geometry: shapely.LineString,
    stop_meters: float
) -> np.ndarray: 
    """
    Each row stores several arrays related to vp.
    vp_direction is an array, vp_idx is an array,
    and the linestring of vp coords can be coerced into an array.
    
    When we're doing nearest neighbor search, we want to 
    first filter the full array down to valid vp
    before snapping it.
    """        
    # These are the valid index values where opposite direction 
    # is excluded       
    valid_indices = (vp_direction_array != opposite_stop_direction).nonzero()   
    
    # These are vp coords where index values of opposite direction is excluded
    valid_vp_coords_array = np.array(vp_geometry.coords)[valid_indices]
    
    # These are the subset of vp_idx values where opposite direction is excluded
    valid_vp_idx_array = np.asarray(vp_idx_array)[valid_indices]  
    
    nearest_indices = find_nearest_points(
        valid_vp_coords_array, 
        stop_geometry, 
        valid_vp_idx_array,
    )
 
    before_vp, after_vp, before_meters, after_meters = filter_to_nearest2_vp(
        valid_vp_coords_array[nearest_indices], # subset of coords in nn
        shape_geometry,
        valid_vp_idx_array[nearest_indices], # subset of vp_idx in nn
        stop_meters,
    )
    
    return before_vp, after_vp, before_meters, after_meters



def grab_vp_timestamp(
    prior_vp: int, 
    subseq_vp: int, 
    vp_idx_array: np.ndarray,
    timestamp_arr: np.ndarray
) -> tuple:
    """
    Need to handle -1
    """
    vp_idx_array = np.asarray(vp_idx_array)
    timestamp_arr = np.asarray(timestamp_arr)
    
    index_of_prior = np.where(vp_idx_array == prior_vp)[0]
    index_of_subseq = np.where(vp_idx_array == subseq_vp)[0]
    
    if index_of_prior.size > 0:
        start_timestamp = timestamp_arr[index_of_prior][0]
    else: 
        start_timestamp = np.nan
    if index_of_subseq.size > 0:
        end_timestamp = timestamp_arr[index_of_subseq][0]
    else:
        end_timestamp = np.nan
    return start_timestamp, end_timestamp


def interpolate_stop_arrival_time(
    stop_position: float, 
    shape_meters_arr: np.ndarray,
    timestamp_arr: np.ndarray
) -> float:
    """
    Interpolate the arrival time given the stop meters position.
    Cast datetimes into floats and cast back as datetime.
    """
    timestamp_arr = np.asarray(timestamp_arr).astype("datetime64[s]").astype("float64")

    return np.interp(
        stop_position, np.asarray(shape_meters_arr), timestamp_arr
    ).astype("datetime64[s]")
    

def convert_timestamp_to_seconds(
    df: pd.DataFrame, 
    timestamp_cols: list,
) -> pd.DataFrame: 
    """
    Convert timestamp into seconds.
    """
    for c in timestamp_cols:
        df = df.assign(
            time_sec = ((df[c].dt.hour * 3_600) + 
                            (df[c].dt.minute * 60) + 
                            (df[c].dt.second)
                       ),
        ).rename(columns = {"time_sec": f"{c}_sec"})
    
    return df

def rolling_window_make_array(
    df: pd.DataFrame, 
    window: int, 
    rolling_col: str
) -> pd.DataFrame:
    """
    Interpolated stop arrival times are checked
    to see if they are monotonically increasing.
    If it isn't, it gets recalculated based on 
    stop_meters (the stop's position) relative to
    other stop arrival times.
    
    https://stackoverflow.com/questions/47482009/pandas-rolling-window-to-return-an-array
    """
    df[f"rolling_{rolling_col}"] = [
        np.asarray(window) for window in 
        df.groupby("trip_instance_key")[rolling_col].rolling(
            window = window, center=True)
    ]
    
    is_monotonic_series = np.vectorize(utils.monotonic_check)(df[f"rolling_{rolling_col}"])
    df[f"{rolling_col}_monotonic"] = is_monotonic_series
    
    return df

def stop_and_arrival_time_arrays_by_trip(
    df: pd.DataFrame, 
    trip_stop_cols: list
) -> pd.DataFrame:
    """
    For stops that violated the monotonically increasing condition,
    set those arrival_times to NaT again.
    Now, look across stops and interpolate again, using stop_meters.
    """
    # Add columns with the trip's stop_meters and arrival_times
    # for only correctly interpolated values
    df_arrays = (df[df.arrival_time.notna()].sort_values(trip_stop_cols)
           .groupby("trip_instance_key", group_keys=False)
           .agg({
               "stop_meters": lambda x: list(x),
               "arrival_time": lambda x: list(x)
           }).rename(columns = {
               "stop_meters": "stop_meters_arr", 
               "arrival_time": "arrival_time_arr"
           }).reset_index()
    )
    
    df2 = pd.merge(
        df,
        df_arrays,
        on = "trip_instance_key",
        how = "inner"
    )

    # Use correct values to fill in the missing arrival times
    df2 = df2.assign(
        arrival_time = df2.apply(
            lambda x: interpolate_stop_arrival_time(
                x.stop_meters, x.stop_meters_arr, x.arrival_time_arr
            ), axis=1
        )
    ).drop(columns = ["stop_meters_arr", "arrival_time_arr"])

    return df2


def enforce_monotonicity_and_interpolate_across_stops(
    df: pd.DataFrame,
    trip_stop_cols: list
) -> pd.DataFrame:
    """
    Do a check to make sure stop arrivals are all monotonically increasing.
    If it fails the check in a window of 3, and the center
    position is not increasing, we will interpolate again using 
    surrounding observations.
    """
    df = convert_timestamp_to_seconds(
        df, ["arrival_time"])

    df = rolling_window_make_array(
        df, 
        window = 3, rolling_col = "arrival_time_sec"
    )
    
    # Subset to trips that have at least 1 obs that violates monotonicity
    trips_with_one_false = (
        df.groupby("trip_instance_key")
        .agg({"arrival_time_sec_monotonic": "min"})
        .reset_index()
        .query('arrival_time_sec_monotonic==0')
        .trip_instance_key
    )
    
    # Set arrival times to NaT if it's not monotonically increasing
    mask = df.arrival_time_sec_monotonic == False 
    df.loc[mask, 'arrival_time'] = np.nan
    
    
    no_fix = df[~df.trip_instance_key.isin(trips_with_one_false)]
    fix1 = df[df.trip_instance_key.isin(trips_with_one_false)]
    fix1 = stop_and_arrival_time_arrays_by_trip(fix1, trip_stop_cols)
    
    drop_me = [
        "arrival_time_sec",
        "rolling_arrival_time_sec", "arrival_time_sec_monotonic"
    ]
    
    fixed_df = pd.concat(
        [no_fix, fix1], axis=0
    ).drop(
        columns = drop_me
    ).sort_values(
        trip_stop_cols
    ).reset_index(drop=True)
        
    return fixed_df


def calculate_speed_from_stop_arrivals(
    df: pd.DataFrame,
    trip_cols: list = ["trip_instance_key"],
    trip_stop_cols: list = ["trip_instance_key", "stop_sequence"]
) -> pd.DataFrame:
    """
    """
    df = convert_timestamp_to_seconds(
        df, ["arrival_time"]
    ).sort_values(trip_stop_cols).reset_index(drop=True)
    
    df = df.assign(
        subseq_arrival_time_sec = (df.groupby(trip_cols, 
                                             observed=True, group_keys=False)
                                  .arrival_time_sec
                                  .shift(-1)
                                 ),
        subseq_stop_meters = (df.groupby(trip_cols, 
                                        observed=True, group_keys=False)
                             .stop_meters
                             .shift(-1)
                            )
    )

    speed = df.assign(
        meters_elapsed = df.subseq_stop_meters - df.stop_meters, 
        sec_elapsed = df.subseq_arrival_time_sec - df.arrival_time_sec,
    )
    
    speed = speed.assign(
        speed_mph = utils.calculate_speed(speed.meters_elapsed, speed.sec_elapsed)
    )
    
    return speed
    
import geopandas as gpd
import numpy as np
import pandas as pd

from update_vars import INPUT_FOLDER, PROJECT_CRS, WGS84

def get_stop_times_with_stop_geometry(analysis_date: str) -> gpd.GeoDataFrame:
    """
    Import stop_times and stops and merge.
    Returns a stop_times table that has stop_geometry attached.
    """
    stop_times = pd.read_parquet(
        f"{INPUT_FOLDER}stop_times_{analysis_date}.parquet",
        columns = [
            "feed_key", "trip_id",
            "stop_id", "stop_sequence",
            "arrival_sec"
        ]
    )
    stops = gpd.read_parquet(
        f"{INPUT_FOLDER}stops_{analysis_date}.parquet",
        columns = ["feed_key", "stop_id", "stop_name", "geometry"]
    ).to_crs(PROJECT_CRS)
    
    df = pd.merge(
        stops,
        stop_times,
        on = ["feed_key", "stop_id"],
        how = "inner"
    )
    
    return df


def condense_by_trip(
    df: gpd.GeoDataFrame,
    group_cols: list = ["feed_key", "trip_id"],
    sort_cols: list = ["feed_key", "trip_id", "stop_sequence"],
    array_cols: list = ["stop_sequence", "geometry"]
) -> gpd.GeoDataFrame:
    """
    Condense stop_times (with stop geometry) to trip grain.
    Save stop_sequence, geometry as lists.
    """
    df2 = (
        df
        .sort_values(sort_cols)
        .groupby(group_cols)
        .agg({
            **{c: lambda x: list(x) for c in array_cols}
        })
        .reset_index()
    )

    return df2 

def find_direction_of_travel(
    gdf: gpd.GeoDataFrame, 
    geometry_col: str = "geometry"
) -> pd.DataFrame:
    """
    Each row is a trip-level array with the stops ordered.
    A stop is compared against the prior stop position
    and stop's primary direction is determined.
    """
    cardinal_direction_series = []
    
    for row in gdf.itertuples():
        current_stop_geom = np.array(getattr(row, "geometry"))
        next_stop_geom = current_stop_geom[1:]
        
        # distance_east, distance_north
        direction_arr = np.asarray(
            # first value is unknown because there is no prior stop to compare to
            ["Unknown"] + 
            [cardinal_definition_rules(pt.x - prior_pt.x, pt.y - prior_pt.y) 
             for pt, prior_pt 
             in zip(next_stop_geom, current_stop_geom)]
        )
        cardinal_direction_series.append(direction_arr)
    
    gdf = gdf.assign(
        stop_primary_direction = cardinal_direction_series
    ).drop(columns = "geometry")
    
    return gdf
    

def cardinal_definition_rules(
    distance_east: float, 
    distance_north: float
) -> str:
    """
    We can determine the primary cardinal direction by looking at the 
    delta_x (distance_east) and delta_y (distance_north).
    From shared_utils.rt_utils
    """
    if abs(distance_east) > abs(distance_north):
        if distance_east > 0:
            return "Eastbound"
        elif distance_east < 0:
            return "Westbound"
        else:
            return "Unknown"
    else:
        if distance_north > 0:
            return "Northbound"
        elif distance_north < 0:
            return "Southbound"
        else:
            return "Unknown"
        
def explode_arrays(
    df: pd.DataFrame,
    array_cols: list = ["stop_sequence", "stop_primary_direction"]
) -> pd.DataFrame:
    """
    Use this to turn trip-level df with columns of arrays
    and explode it so it's a long df.
    """
    return df.explode(array_cols)
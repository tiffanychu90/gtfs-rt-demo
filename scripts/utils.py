"""
Utility functions
"""
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from scipy.spatial import KDTree
from typing import Union

import create_table
from update_vars import INPUT_FOLDER, OUTPUT_FOLDER, PROJECT_CRS, WGS84

MPH_PER_MPS = 2.237  # use to convert meters/second to miles/hour

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
    geometry_col: str = "geometry",
    array_cols: list = ["stop_sequence"]
) -> gpd.GeoDataFrame:
    """
    Condense stop_times (with stop geometry) to trip grain.
    Save stop_sequence, geometry as lists.
    """
    orig_crs = df.crs.to_epsg()
    
    df2 = (
        df
        .sort_values(sort_cols)
        .groupby(group_cols)
        .agg({
            geometry_col: lambda x: shapely.LineString(list(x)),
            **{c: lambda x: list(x) for c in array_cols}
        })
        .reset_index()
    )
    
    gdf = gpd.GeoDataFrame(df2, geometry=geometry_col, crs=orig_crs)

    return gdf 


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


def scheduled_and_vp_trips(analysis_date: str) -> list:
    """
    """
    scheduled_trips = pd.read_parquet(
        f"{OUTPUT_FOLDER}trips_{analysis_date}.parquet",
        columns = ["trip_instance_key"] 
    ).trip_instance_key.unique()

    rt_trips = pd.read_parquet(
        f"{OUTPUT_FOLDER}vp_{analysis_date}.parquet",
        columns = ["trip_instance_key"]
    ).trip_instance_key.unique()
    
    
    return list(set(scheduled_trips).intersection(rt_trips))


def plot_vp_shape_stops(
    vp: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    vp_as_line: bool = True
) -> folium.Map:
    """
    vp: raw vp that's long will get condensed into a linestring path
    shapes: shapes linestring
    stops: any stop gdf, either stops or stop_times_direction
    """    
    if vp_as_line:
        vp_condensed = condense_by_trip(
            vp.to_crs(WGS84),
            group_cols = ["trip_id"],
            sort_cols = ["trip_id", "location_timestamp_local"],
            geometry_col = "geometry",
            array_cols = ["location_timestamp_local"]
        )
    else:
        vp_condensed = vp.to_crs(WGS84)

    m = vp_condensed.drop(columns = "location_timestamp_local").explore(
        vp_condensed.index,
        tiles = "CartoDB Positron", 
        categorical=True, legend=False, name = "vp"
    )

    m = stops.to_crs(WGS84).explore(
        "stop_sequence", m=m, categorical=True, legend=False,
        name="stops"
    )

    m = shapes[["shape_id", "geometry"]].to_crs(WGS84).explore(
        "shape_id", color = "orange", name="Shape",
        m=m
    )

    folium.LayerControl().add_to(m)

    return m


def calculate_speed(meters_elapsed: float, sec_elapsed: float) -> float:
    return meters_elapsed / sec_elapsed * MPH_PER_MPS

def monotonic_trips(
    analysis_date: str,
    **kwargs
):
    """
    """
    stops_projected = create_table.stop_times_projected_table(
        analysis_date, 
        **kwargs
    )
    
    trip_cols = ["schedule_gtfs_dataset_key", "trip_id", "shape_id"]

    check_df = (stops_projected
                       .sort_values(trip_cols + ["stop_sequence"])
                       .groupby(trip_cols)
                       .agg({"stop_meters": lambda x: list(x)})
                       .reset_index()
                      )

    check_df = check_df.assign(
        is_monotonic = check_df.apply(lambda x: np.all(np.diff(x.stop_meters) > 0), axis=1)
    )[
        trip_cols + ["is_monotonic"]
    ].drop_duplicates().reset_index(drop=True)

    return check_df



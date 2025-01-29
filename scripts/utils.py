"""
Utility functions
"""
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from typing import Literal, Union

import create_table
from update_vars import (OUTPUT_FOLDER, PROJECT_CRS, WGS84,
                         gtfs_tables_list)

MPH_PER_MPS = 2.237  # use to convert meters/second to miles/hour

OPERATOR_NAMES_DICT = {
    "1fd2f07342d966919b15d5d37fda8cc8": "Bruin Bus",
    "efbbd5293be71f7a5de0cf82b59febe1": "Big Blue Bus",
    "364d59b3aea55aec2962a0b3244a40e0": "Santa Clarita",
    "cf0f7df88da36cd9ca4248eb1d6a0f39": "Culver City Bus",
    "cc53a0dbf5df90e3009b9cb5d89d80ba": "LADOT"
}

def condense_by_trip(
    df: gpd.GeoDataFrame,
    group_cols: list = ["schedule_gtfs_dataset_key", "trip_instance_key"],
    sort_cols: list = ["schedule_gtfs_dataset_key", "trip_instance_key", "stop_sequence"],
    geometry_col: str = "geometry",
    array_cols: list = ["stop_sequence"]
) -> gpd.GeoDataFrame:
    """
    Condense stop_times or vp to trip grain.
    Get points strung together as linestring path.
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
        

def scheduled_and_vp_trips() -> list:
    """
    Get list of trip_instance_keys that are in common from 
    scheduled and vp.
    """
    scheduled_trips = pd.read_parquet(
        f"{OUTPUT_FOLDER}trips.parquet",
        columns = ["trip_instance_key"] 
    ).trip_instance_key.unique()

    rt_trips = pd.read_parquet(
        f"{OUTPUT_FOLDER}vp.parquet",
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
    """
    Convert meters and seconds elapsed into speed_mph.
    """
    return meters_elapsed / sec_elapsed * MPH_PER_MPS


def monotonic_check(arr: np.ndarray) -> bool:
    """
    For an array, check if it's monotonically increasing. 
    https://stackoverflow.com/questions/4983258/check-list-monotonicity
    """
    diff_arr = np.diff(arr)
    
    if np.all(diff_arr > 0):
        return True
    else:
        return False

    
def monotonic_trips(
    stop_times_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Based on stop_meters, the stop's point geometry
    projected against shape's line geometry,
    we can find which trips do not have monotonically
    increasing stops.
    
    This is the main shortcoming with simply projecting
    points on the line, because numbers jump around,
    reflecting a pattern in how buses travel on roads,
    which isn't wrong, but we need a method that handles this.
    """
    trip_cols = ["schedule_gtfs_dataset_key", "trip_id", "shape_id"]

    gdf = (stop_times_gdf
               .sort_values(trip_cols + ["stop_sequence"])
               .groupby(trip_cols)
               .agg({"stop_meters": lambda x: list(x)})
               .reset_index()
              )

    gdf = gdf.assign(
        is_monotonic = gdf.apply(lambda x: monotonic_check(x.stop_meters), axis=1)
    )[trip_cols + ["is_monotonic"]
    ].drop_duplicates().reset_index(drop=True)

    return gdf



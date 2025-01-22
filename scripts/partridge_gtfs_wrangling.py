"""
Use gtfs_segments.partridge module to download GTFS feeds gtfs.zip
for operators.

Set up functions to do preprocessing for stop_times grain table.
"""
import datetime
import geopandas as gpd
import gtfs_segments
import numpy as np
import pandas as pd

import utils
from update_vars import PARTRIDGE_FOLDER, PROJECT_CRS 

def get_stop_times_with_stop_geometry(
    operator_name: str
) -> gpd.GeoDataFrame:
    """
    For feed downloaded from partridge, 
    combine stop_times, stops, shapes, trips
    and get preprocessed table.
    """
    stop_times = pd.read_parquet(
        f"{PARTRIDGE_FOLDER}{operator_name}/stop_times.parquet",
        columns = [
            "trip_id",
            "stop_id", "stop_sequence",
            "arrival_time"
        ]
    ).rename(columns = {"arrival_time": "arrival_sec"})
    
    stops = gpd.read_parquet(
        f"{PARTRIDGE_FOLDER}{operator_name}/stops.parquet",
        columns = ["stop_id", "stop_name", "geometry"]
    ).to_crs(PROJECT_CRS)
    
    trips = pd.read_parquet(
        f"{PARTRIDGE_FOLDER}{operator_name}/trips.parquet",
        columns = [
            "trip_id", "shape_id",
        ]
    )
    
    shapes = gpd.read_parquet(
        f"{PARTRIDGE_FOLDER}{operator_name}/shapes.parquet",
        columns = ["shape_id", "geometry"]
    ).to_crs(PROJECT_CRS)
    
    gdf = merge_stop_times_trips_shapes_stops(
        stop_times,
        stops,
        trips,
        shapes,
        stop_group = ["stop_id"],
        trip_group = ["trip_id"],
        shape_group = ["shape_id"]
    )
    
    gdf2 = stop_times_preprocessing(gdf, trip_group = ["trip_id"])
    
    return gdf2


def merge_stop_times_trips_shapes_stops(
    stop_times_df: pd.DataFrame,
    stops_gdf: gpd.GeoDataFrame,
    trips_df: pd.DataFrame,
    shapes_gdf: gpd.GeoDataFrame,
    stop_group: list = ["stop_id"],
    trip_group: list = ["trip_id"],
    shape_group: list = ["shape_id"]
) -> gpd.GeoDataFrame:
    """
    Combine all 4 GTFS schedule tables: 
    stop_times, stops, shapes, trips.
    
    We need a stop_time grain table.
    Attach stop geometry and shape geometry 
    (using trips to link trips to shapes).
    
    The merge columns, stop_group, trip_group, shape_group,
    here are defined for an individual operator.
    If a df with multiple operators is used, these should be 
    defined as trip_group = [operator_identifier, 'trip_id'] 
    since trip_id, stop_id, shape_id are not necessarily unique across operators.
    """    
    gdf = pd.merge(
        stops_gdf,
        stop_times_df,
        on = stop_group,
        how = "inner"
    ).merge(
        trips_df,
        on = trip_group,
        how = "inner"
    ).merge(
        shapes_gdf.rename(columns = {"geometry": "shape_geometry"}),
        on = shape_group,
        how = "inner"
    ).sort_values(trip_group + ["stop_sequence"]).reset_index(drop=True)
        
    return gdf


def stop_times_preprocessing(
    gdf: gpd.GeoDataFrame,
    trip_group: list = ["trip_id"]
) -> gpd.GeoDataFrame:
    """
    All the stuff we want to do to stop_times + shapes + stops + trips.
    
    For stop_times, we want to add columns to understand:
    - stop_primary_direction: direction from prior stop
    - stop_meters: the stop's point geometry projected against the shape geometry 
    (meters progressed along shape)
    - stop_seq_pair: a segment can be defined as a pair of stop_sequences
    - stop_id_pair: a segment can be defined as a pair of stop_ids
    
    stop_seq_pair is more intuitive to check what's happening,
    but stop_id_pair will help us aggregate speeds across many trips
    """
    prior_geometry = (gdf
                  .groupby(trip_group, group_keys=False)
                  .geometry
                  .shift(1)
                )    
    
    gdf = gdf.assign(
        stop_primary_direction = np.vectorize(utils.cardinal_definition_rules)(
            gdf.geometry.x - prior_geometry.x, 
            gdf.geometry.y - prior_geometry.y),
        stop_meters = gdf.shape_geometry.project(gdf.geometry)
    )
    
    gdf = gdf.assign(
        subseq_stop_sequence = (gdf
                           .groupby(trip_group, group_keys=False)
                           .stop_sequence
                           .shift(-1)
                          ).astype("Int64"),
        stop_id2 = (gdf
                    .groupby(trip_group, group_keys=False)
                    .stop_id
                    .shift(-1)
                    ),
        subseq_stop_meters = (gdf
                   .groupby(trip_group, group_keys=False)
                   .stop_meters
                   .shift(-1)
                  ),
    ).rename(columns = {"stop_id": "stop_id1"})

    gdf = gdf.assign(
        stop_seq_pair = gdf.stop_sequence.astype(str).str.cat(
            gdf.subseq_stop_sequence.astype(str), 
            sep="__"
        ),
        stop_id_pair = gdf.stop_id1.str.cat(gdf.stop_id2, sep="__")
    ).drop(
        columns = ["subseq_stop_sequence", "shape_geometry"]
    )
    
    return gdf


def vp_preprocessing(
    gdf: gpd.GeoDataFrame,
    trip_group: list = ["trip_id"]
) -> gpd.GeoDataFrame:
    """
    All the stuff we want to do to vehicle_positions.
    
    For vp, we want to add columns to understand:
    - vp_primary_direction: direction from prior stop
    - vp_meters: the vp's point geometry projected against the shape geometry 
    (meters progressed along shape)
    We should also get vp dwell positions, 
    and have location_timestamp_local and moving_timestamp_local.
    """  
    prior_geometry = (gdf
                  .groupby(trip_group, group_keys=False)
                  .geometry
                  .shift(1)
                )    
    
    gdf = gdf.assign(
        vp_primary_direction = np.vectorize(utils.cardinal_definition_rules)(
            gdf.geometry.x - prior_geometry.x, 
            gdf.geometry.y - prior_geometry.y),
        vp_meters = gdf.shape_geometry.project(gdf.geometry),
        vp_idx = gdf.index, # it's ordered within a trip, but vp_idx spans entirety of vp
    )
    
    if "feed_key" in gdf.columns:
        gdf = gdf.drop(columns = "feed_key") 
    
    return gdf
    
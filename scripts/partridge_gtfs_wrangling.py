"""
"""
import datetime
import geopandas as gpd
import gtfs_segments
import numpy as np
import pandas as pd

import utils
from update_vars import PARTRIDGE_FOLDER, WGS84, PROJECT_CRS 

def get_stop_times_with_stop_geometry(
    operator_name: str
) -> gpd.GeoDataFrame:
    """
    Import stop_times and stops and merge.
    Returns a stop_times table that has stop_geometry attached.
    """
    stop_times = pd.read_parquet(
        f"{PARTRIDGE_FOLDER}{operator_name}/stop_times.parquet",
        columns = [
            "trip_id",
            "stop_id", "stop_sequence",
            "arrival_time"
        ]
    )
    
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
    
    df = pd.merge(
        stops,
        stop_times,
        on = "stop_id",
        how = "inner"
    ).merge(
        trips,
        on = "trip_id",
        how = "inner"
    ).merge(
        shapes.rename(columns = {"geometry": "shape_geometry"}),
        on = "shape_id",
        how = "inner"
    ).sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)
    
    return df


def stop_times_preprocessing(
    gdf: gpd.GeoDataFrame,
    trip_group: list = ["trip_id"]
) -> gpd.GeoDataFrame:
    """
    All the stuff we want to do to stop_times + shapes + stops + trips
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
    ).to_crs(WGS84)
    
    return gdf

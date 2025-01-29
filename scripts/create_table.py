"""
Create tables.

Mainly we need a stop_time grain table with scheduled data
to be used with trip grain vehicle positions data 
for segment speeds calculation.
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from typing import Literal, Union

import partridge_gtfs_wrangling
import neighbor
import utils
from update_vars import (OUTPUT_FOLDER,
                         gtfs_tables_list, 
                         PROJECT_CRS
                        )


def get_hackathon_table_filepath(
    table_name: Literal[gtfs_tables_list] = "", 
    folder_path: str = OUTPUT_FOLDER
) -> str:
    """
    Get local filepaths for each of the 
    hackathon-prepped Cal-ITP tables.
    """
    return f"{folder_path}{table_name}.parquet"


def get_calitp_table(
    table_name: Literal[gtfs_tables_list] = "", 
    folder_path: str = OUTPUT_FOLDER,
    **kwargs
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Import any of the 6 available GTFS tables.
    stop_times_direction is a combination of stop_times + trips (route info) + stop (geometry)
    """
    if table_name in ["trips", "stop_times"]:                

        df = pd.read_parquet(
            get_hackathon_table_filepath(table_name, folder_path), 
            **kwargs
        )
        
    elif table_name in ["stops", "shapes", "stop_times_direction", "vp", "segments"] :
   
        df = gpd.read_parquet(
            get_hackathon_table_filepath(table_name, folder_path),
            **kwargs
        )
        
    return df.drop_duplicates().reset_index(drop=True)

        
def stop_times_projected_calitp_table(
    crs: str = PROJECT_CRS,
    trip_group: list = ["service_date", "trip_id"],
    folder_path: str = OUTPUT_FOLDER,
    **trip_kwargs,
) -> gpd.GeoDataFrame:
    """
    Get stop_times_direction table (stop_times + stops)
    and project each stop position against shape geometry.
    """
    operator_trip_group = ["schedule_gtfs_dataset_key"] + trip_group
    
    trips = get_calitp_table(
        "trips", 
        folder_path = folder_path,
        columns =  operator_trip_group + ["trip_instance_key", "shape_id"],
        **trip_kwargs
    )

    # trip_id can be repeated across operators
    # Once we move out of single operator, we use trip_instance_key / shape_array_key,
    # which is present in our warehouse but would have to be created for a tool
    subset_trips = trips.trip_id.unique().tolist()
    subset_shapes = trips.shape_id.unique().tolist()
    
    stop_times = get_calitp_table(
        "stop_times", 
        folder_path = folder_path,
        filters = [[("trip_id", "in", subset_trips)]],
        columns = operator_trip_group + ["stop_id", "stop_sequence"]
    )
    
    stops = get_calitp_table(
        "stops",
        folder_path = folder_path,
        filters = [[("stop_id", "in", stop_times.stop_id)]],
        columns = ["schedule_gtfs_dataset_key", "service_date", "stop_id", "stop_name", "geometry"]
    ).to_crs(crs)
    
    shapes = get_calitp_table(
        "shapes",
        folder_path = folder_path,
        filters = [[("shape_id", "in", subset_shapes)]],
        columns = ["schedule_gtfs_dataset_key", "service_date", "shape_id", "geometry"]
    ).to_crs(crs)
    
    # Project each stop onto shape
    gdf = partridge_gtfs_wrangling.merge_stop_times_trips_shapes_stops(
        stop_times,
        stops,
        trips,
        shapes,
        stop_group = ["schedule_gtfs_dataset_key", "service_date", "stop_id"],
        trip_group = operator_trip_group,
        shape_group = ["schedule_gtfs_dataset_key", "service_date", "shape_id"]
    )
    
    # Get stop_pair and other  (we use stop_pair), but for simplicity, 
    # this illustrates what we do anyway    
    gdf = partridge_gtfs_wrangling.stop_times_preprocessing(
        gdf,
        trip_group = operator_trip_group + ["trip_instance_key"]
    )

    return gdf

    
def vp_projected_table(
    crs: str = PROJECT_CRS,
    folder_path: str = OUTPUT_FOLDER,
    **trip_kwargs,
) -> gpd.GeoDataFrame:
    """
    Get vp table
    and project each vehicle position against shape geometry.
    """        
    trip_cols = ["service_date", "schedule_gtfs_dataset_key", "trip_instance_key", "trip_id"]              
    
    # To merge with vp, we need to use `schedule_gtfs_dataset_key`
    # In the Cal-ITP warehouse, feed_key is used to merge schedule tables
    # and `schedule_gtfs_dataset_key` is present only in trips, used to merge operators from other RT tables
    trips = get_calitp_table(
        "trips", 
        folder_path = folder_path,
        columns = trip_cols + ["shape_id"],
        **trip_kwargs
    )
    
    # trip_id can be repeated across operators
    # Once we move out of single operator, we use trip_instance_key / shape_array_key,
    # which is present in our warehouse but would have to be created for a tool
    subset_trips = trips.trip_instance_key.unique().tolist()
    subset_shapes = trips.shape_id.unique().tolist()
    
    # vp would not have feed_key, it always has to be keyed with schedule_gtfs_dataset_key
    vp = get_calitp_table(
        "vp",
        folder_path = folder_path,
        filters = [[("trip_instance_key", "in", subset_trips)]],
        columns = trip_cols + ["location_timestamp_local", "geometry"]
    ).to_crs(crs).sort_values(
        "location_timestamp_local"
    ).reset_index(drop=True)
    
    shapes = get_calitp_table(
        "shapes",
        folder_path = folder_path,
        filters = [[("shape_id", "in", subset_shapes)]],
        columns = ["schedule_gtfs_dataset_key", "shape_id", "geometry"]
    ).to_crs(crs)
    
    trips_to_shape = pd.merge(
        shapes,
        trips,
        on = ["schedule_gtfs_dataset_key", "shape_id"],
        how = "inner"
    )
    
    gdf = pd.merge(
        vp,
        trips_to_shape.rename(columns = {"geometry": "shape_geometry"}),
        on = trip_cols,
        how = "inner"
    )  
    
    gdf = partridge_gtfs_wrangling.vp_preprocessing(
        gdf, 
        trip_group = trip_cols
    )
    
    return gdf


def stop_times_with_vp_table(
    **kwargs
) -> gpd.GeoDataFrame:
    """
    Put together stop times with direction with vp.
    This is our processed df ready for deriving speeds.
    """
    # We created this in stop_times_direction.py
    stops_projected = get_calitp_table(
        "stop_times_direction", 
        folder_path = OUTPUT_FOLDER,
        **kwargs
    )

    vp_projected = vp_projected_table(
        crs = PROJECT_CRS,
        folder_path = OUTPUT_FOLDER,
        **kwargs
    )  
    
    vp_nn = utils.condense_by_trip(
        vp_projected,
        group_cols = ["service_date", "trip_instance_key", "shape_geometry"],
        sort_cols = ["service_date", "trip_instance_key", "vp_idx"],
        geometry_col = "geometry",
        array_cols = ["vp_idx", "vp_primary_direction", "location_timestamp_local"]
    )
    
    vp_nn = vp_nn.assign(
        vp_primary_direction = vp_nn.apply(
            lambda x: np.array(x.vp_primary_direction), 
            axis=1),
        vp_idx = vp_nn.apply(lambda x: np.array(x.vp_idx), axis=1)
    )
        
    gdf = pd.merge(
        stops_projected.rename(columns = {"geometry": "stop_geometry"}),
        vp_nn.rename(columns = {"geometry": "vp_geometry"}),
        on = ["service_date", "trip_instance_key"],
        how = "inner"
    ).set_geometry("stop_geometry")
    
    gdf = gdf.assign(
        stop_opposite_direction = gdf.stop_primary_direction.map(neighbor.OPPOSITE_DIRECTIONS),
    )
    
    return gdf
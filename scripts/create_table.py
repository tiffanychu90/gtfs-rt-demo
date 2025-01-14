"""
Create tables for speed calculation.
"""
import geopandas as gpd
import pandas as pd
import shapely

from typing import Literal, Union

from update_vars import OUTPUT_FOLDER, analysis_date, PROJECT_CRS, gtfs_tables_list

# This does include a key that identifies the transit operator
# Switch to natural identifier trip_id, instead of trip_instance_key (which links schedule to RT tables in our warehouse)
trip_group = ["schedule_gtfs_dataset_key", "trip_id"]

def get_table_filepath(
    table_name: Literal[gtfs_tables_list], 
    analysis_date: str
) -> str:
    """
    Get local filepaths for each of the tables.
    """
    return f"{OUTPUT_FOLDER}{table_name}_{analysis_date}.parquet"


def get_table(
    table_name: Literal[gtfs_tables_list], 
    analysis_date: str, 
    **kwargs
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Import any of the 6 available GTFS tables.
    stop_times_direction is a combination of stop_times + trips (route info) + stop (geometry)
    """
    if table_name in ["trips", "stop_times"]:
        df = pd.read_parquet(
            get_table_filepath(table_name, analysis_date), 
            **kwargs
        )
        
    if table_name in ["stops", "shapes", "stop_times_direction", "vp"]:
        df = gpd.read_parquet(
            get_table_filepath(table_name, analysis_date), 
            **kwargs
        )   
        
    return df.drop_duplicates().reset_index(drop=True)

        
def stop_times_projected_table(
    analysis_date, 
    crs: str = PROJECT_CRS,
    **trip_kwargs,
) -> gpd.GeoDataFrame:
    """
    Get stop_times_direction table (stop_times + stops)
    and project each stop position against shape geometry.
    """
    trips = get_table(
        "trips", 
        analysis_date, 
        columns = trip_group + ["shape_id"],
        **trip_kwargs
    )
    
    # trip_id can be repeated across operators
    # Once we move out of single operator, we use trip_instance_key / shape_array_key,
    # which is present in our warehouse but would have to be created for a tool
    subset_trips = trips.trip_id.unique().tolist()
    subset_shapes = trips.shape_id.unique().tolist()
    
    stop_times_direction = get_table(
        "stop_times_direction", 
        analysis_date,
        filters = [[("trip_id", "in", subset_trips)]],
        columns = trip_group + ["stop_id", "stop_sequence", "geometry"]
    ).to_crs(crs)

    shapes = get_table(
        "shapes",
        analysis_date,
        filters = [[("shape_id", "in", subset_shapes)]],
        columns = ["schedule_gtfs_dataset_key", "shape_id", "geometry"]
    ).to_crs(crs)
    
    # Project each stop onto shape
    gdf = pd.merge(
        stop_times_direction,
        trips,
        on = trip_group,
        how = "inner"
    ).merge(
        shapes.rename(columns = {"geometry": "shape_geometry"}),
        on = ["schedule_gtfs_dataset_key", "shape_id"],
        how = "inner"
    )
    
    # Get stop_seq_pair (we use stop_pair), but for simplicity, 
    # this illustrates what we do anyway
    gdf = gdf.assign(
        stop_meters = gdf.shape_geometry.project(gdf.geometry),
        subseq_stop_sequence = (gdf
                           .sort_values(trip_group + ["stop_sequence"])
                           .groupby(trip_group, group_keys=False)
                           .stop_sequence
                           .shift(-1)
                          ).astype("Int64"),
        stop_id2 = (gdf
                    .sort_values(trip_group + ["stop_sequence"])
                    .groupby(trip_group, group_keys=False)
                    .stop_id
                    .shift(-1)
                    ),
    )
    
    gdf = gdf.assign(
        stop_seq_pair = gdf.stop_sequence.astype(str).str.cat(
            gdf.subseq_stop_sequence.astype(str), 
            sep="__"
        )
    ).drop(columns = ["subseq_stop_sequence", "shape_geometry"])
    
    
    return gdf


def vp_projected_table(
    analysis_date: str, 
    crs: str = PROJECT_CRS,
    **trip_kwargs,
) -> gpd.GeoDataFrame:
    """
    Get vp table
    and project each vehicle position against shape geometry.
    """
    trips = get_table(
        "trips", 
        analysis_date, 
        columns = trip_group + ["shape_id"],
        **trip_kwargs
    )
    
    # trip_id can be repeated across operators
    # Once we move out of single operator, we use trip_instance_key / shape_array_key,
    # which is present in our warehouse but would have to be created for a tool
    subset_trips = trips.trip_id.unique().tolist()
    subset_shapes = trips.shape_id.unique().tolist()
    
    vp = get_table(
        "vp",
        analysis_date,
        filters = [[("trip_id", "in", subset_trips)]],
        columns = trip_group + ["location_timestamp_local", "geometry"]
    ).to_crs(crs).sort_values(
        "location_timestamp_local"
    ).reset_index(drop=True)
    
    shapes = get_table(
        "shapes",
        analysis_date,
        filters = [[("shape_id", "in", subset_shapes)]],
        columns = ["schedule_gtfs_dataset_key", "shape_id", "geometry"]
    ).to_crs(crs)
    
    gdf = pd.merge(
        vp,
        trips,
        on = trip_group,
        how = "inner"
    ).merge(
        shapes.rename(columns = {"geometry": "shape_geometry"}),
        on = ["schedule_gtfs_dataset_key", "shape_id"],
        how = "inner"
    )

    gdf = gdf.assign(
        vp_meters = gdf.shape_geometry.project(gdf.geometry),
        vp_idx = gdf.index, # it's ordered within a trip, but vp_idx spans entirety of vp
    ).drop(columns = ["shape_geometry"])
    
    return gdf
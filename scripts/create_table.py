"""
Create tables for speed calculation.
"""
import geopandas as gpd
import pandas as pd
import shapely

from typing import Literal, Union

from update_vars import OUTPUT_FOLDER, analysis_date, PROJECT_CRS, gtfs_tables_list

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
        columns = ["trip_id", "shape_id"],
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
        columns = ["trip_id", "stop_sequence", "geometry"]
    ).to_crs(crs)

    shapes = get_table(
        "shapes",
        analysis_date,
        filters = [[("shape_id", "in", subset_shapes)]],
        columns = ["shape_id", "geometry"]
    ).to_crs(crs)
    
    # Project each stop onto shape
    gdf = pd.merge(
        stop_times_direction,
        trips,
        on = "trip_id",
        how = "inner"
    ).merge(
        shapes.rename(columns = {"geometry": "shape_geometry"}),
        on = "shape_id",
        how = "inner"
    )
    
    gdf = gdf.assign(
        stop_meters = gdf.shape_geometry.project(gdf.geometry),
    ).drop(columns = ["shape_geometry"])
    
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
        columns = ["trip_id", "shape_id"],
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
        columns = ["trip_id", "location_timestamp_local", "geometry"]
    ).to_crs(crs).sort_values(
        "location_timestamp_local"
    ).reset_index(drop=True)
    
    shapes = get_table(
        "shapes",
        analysis_date,
        filters = [[("shape_id", "in", subset_shapes)]],
        columns = ["shape_id", "geometry"]
    ).to_crs(crs)
    
    gdf = pd.merge(
        vp,
        trips,
        on = "trip_id",
        how = "inner"
    ).merge(
        shapes.rename(columns = {"geometry": "shape_geometry"}),
        on = "shape_id",
        how = "inner"
    )

    gdf = gdf.assign(
        vp_meters = gdf.shape_geometry.project(gdf.geometry)
    ).drop(columns = ["shape_geometry"])
    
    return gdf
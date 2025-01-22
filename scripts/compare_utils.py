import branca
import geopandas as gpd
import numpy as np
import pandas as pd

def speed_stats(gdf):
    min_speed = gdf.speed_mph.min()
    max_speed = gdf.speed_mph.max()
    
    speeds_calculated = gdf[
        (gdf.speed_mph.notna()) & 
        (gdf.speed_mph < np.inf)
    ]
    
    print(f"min speed: {min_speed}, max speed: {max_speed}")
    print(f"total rows: {len(gdf)}")
    print(f"rows with invalid speeds: {len(gdf) - len(speeds_calculated)}")
    print(f"rows with valid speeds: {len(speeds_calculated)}")
    
    display(speeds_calculated.speed_mph.hist(bins=range(0, 80, 5)))
    return

def speed_map(speed_gdf: gpd.GeoDataFrame):
    COLORSCALE = branca.colormap.step.RdBu_10.scale(vmin=0, vmax=80)
    drop_cols = speed_gdf.select_dtypes("datetime").columns
    
    m = speed_gdf[(
        speed_gdf.speed_mph.notna()) & 
        (speed_gdf.speed_mph < np.inf)
    ].drop(
        columns = drop_cols
    ).set_geometry(
        "segment_geometry"
    ).explore(
        "speed_mph", cmap=COLORSCALE,
        tiles = "CartoDB Positron"
    )
    return m 
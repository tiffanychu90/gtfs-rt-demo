import datetime
import geopandas as gpd
import pandas as pd

import create_table
from update_vars import OUTPUT_FOLDER, PROJECT_CRS

if __name__ == "__main__":
    
    start = datetime.datetime.now()
    
    gdf = create_table.stop_times_projected_calitp_table( 
        crs = PROJECT_CRS,
        folder_path = OUTPUT_FOLDER,
    )
    
    gdf.to_parquet(f"{OUTPUT_FOLDER}stop_times_direction.parquet")    
    
    end = datetime.datetime.now()
    print(f"execution time: {end - start}")
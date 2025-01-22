import datetime
import geopandas as gpd
import pandas as pd

import create_table
from update_vars import INPUT_FOLDER, analysis_date, PROJECT_CRS

if __name__ == "__main__":
    
    start = datetime.datetime.now()
    
    gdf = create_table.stop_times_projected_calitp_table(
        analysis_date, 
        crs = PROJECT_CRS,
        trip_group = ["trip_id"],
        folder_path = INPUT_FOLDER,
    )
    
    gdf.to_parquet(f"{INPUT_FOLDER}stop_times_direction_{analysis_date}.parquet")    
    
    end = datetime.datetime.now()
    print(f"execution time: {end - start}")
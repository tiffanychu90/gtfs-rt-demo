"""
Download 2 operators, save GTFS schedule tables.
Also pre-process stop_times.
"""
import gtfs_segments
import os

import partridge_gtfs_wrangling
from update_vars import PARTRIDGE_FOLDER, operators_list

def export_schedule_parquets(
    provider_name: str,
    readable_name: str,
    input_path: str,
    export_path: str,
):
    """
    Export trips, shapes, stops, stop_times 
    GTFS schedule tables from gtfs.zip.
    We'll use this to help with our preprocessing steps.
    """
    if not os.path.exists(export_path):
        os.makedirs(export_path)
        
    feed = gtfs_segments.partridge_func.get_bus_feed(
        f"{input_path}/gtfs.zip"
    )

    trips_df = feed[1].trips 
    shapes_df = feed[1].shapes
    stops_df = feed[1].stops
    stop_times_df = feed[1].stop_times
    
    trips_df.to_parquet(f"{export_path}/trips.parquet")
    shapes_df.to_parquet(f"{export_path}/shapes.parquet")
    stops_df.to_parquet(f"{export_path}/stops.parquet")
    stop_times_df.to_parquet(f"{export_path}/stop_times.parquet")

    segments = gtfs_segments.gtfs_segments.process_feed(feed[1])
    segments.to_parquet(f"{export_path}/segments.parquet")
    
    return

if __name__ == "__main__":
    
    for readable_name in operators_list:
        
        print(f"Downloading {readable_name}")
        sources_df = gtfs_segments.fetch_gtfs_source(place=readable_name)
        
        provider_name = sources_df.provider.iloc[0]
    
        gtfs_segments.mobility.download_latest_data(sources_df, PARTRIDGE_FOLDER)

        export_schedule_parquets(
            provider_name = provider_name,
            readable_name = readable_name,
            input_path = f"{PARTRIDGE_FOLDER}{provider_name}",
            export_path = f"{PARTRIDGE_FOLDER}{readable_name}"
        )
        
        print(f"Exported {provider_name} as {readable_name}")

        stop_times_direction = partridge_gtfs_wrangling.get_stop_times_with_stop_geometry(
            readable_name
        ).pipe(
            partridge_gtfs_wrangling.stop_times_preprocessing, 
            trip_group = ["trip_id"]
        )
        
        stop_times_direction.to_parquet(
            f"{PARTRIDGE_FOLDER}{readable_name}/stop_times_direction.parquet"
        )
        print(f"stop times preprocessing ")

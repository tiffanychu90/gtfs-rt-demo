"""
Download one operator's feed, instead of our warehouse
and get segments from it.
"""
import gtfs_segments

from update_vars import PARTRIDGE_FOLDER

if __name__ == "__main__":
    one_operator = "LADOT"
    
    sources_df = gtfs_segments.mobility.fetch_gtfs_source(place=one_operator)
    
    FEED_PROVIDER = sources_df.provider.iloc[0]
    
    gtfs_segments.mobility.download_latest_data(sources_df, PARTRIDGE_FOLDER)

    df = gtfs_segments.gtfs_segments.get_gtfs_segments(
        f'{PARTRIDGE_FOLDER}{FEED_PROVIDER}/gtfs.zip'
    )
    df.to_parquet(f"{PARTRIDGE_FOLDER}{one_operator}_segments.parquet")
    print(f"saved: {one_operator} segments")

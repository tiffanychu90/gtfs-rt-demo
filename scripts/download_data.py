"""
Download one operator's feed, instead of our warehouse
and get segments from it.
"""
import gtfs_segments
import pyaml 

import gtfs_segments

from update_vars import PARTRIDGE_FOLDER

def create_operators_yaml(
    state_abbrev: str = "CA",
    yaml_file: str = f"{PARTRIDGE_FOLDER}transit_operators.yml"
):
    sources_df = gtfs_segments.mobility.fetch_gtfs_source(place=f"-{state_abbrev}")

    sources_df = sources_df.assign(
        #place = sources_df.provider.str.split("-", expand=True)[0],
        name = sources_df.provider.str.split("-", expand=True)[1].str.replace(f"-{state_abbrev}", "")
    )
    

    my_dict = {
        **{
            readable_name: provider 
            for provider, readable_name 
            in zip(sources_df.provider, sources_df.name)
          }
    }  
    
    output = pyaml.dump(my_dict, sort_keys=False)

    with open(yaml_file, "w") as f:
        f.write(output)
    
    print(f"exported {yaml_file}")
    
    return


if __name__ == "__main__":
    
    create_operators_yaml()
    '''
    one_operator = "LADOT"
    
    sources_df = gtfs_segments.mobility.fetch_gtfs_source(place="CA")
    
    FEED_PROVIDER = sources_df.provider.iloc[0]
    
    gtfs_segments.mobility.download_latest_data(sources_df, PARTRIDGE_FOLDER)

    df = gtfs_segments.gtfs_segments.get_gtfs_segments(
        f'{PARTRIDGE_FOLDER}{FEED_PROVIDER}/gtfs.zip'
    )
    df.to_parquet(f"{PARTRIDGE_FOLDER}{one_operator}_segments.parquet")
    print(f"saved: {one_operator} segments")
    '''

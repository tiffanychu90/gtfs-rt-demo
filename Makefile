install_env:
	pip install -r requirements.txt
        
process_data:
	python scripts/stop_times_direction.py
	python scripts/subset_tables.py

# This didn't work, but moving to git lfs did
# git lfs install # add .gitattributes file after this
# git lfs track “full_data/*.parquet”
# git add “full_data/*.parquet”
# git lfs ls-files # to see if full_data is added
# git lfs track "sample_data/*.parquet"
# git add full_data/*.parquet
# Add these to .gitattributes (use the quotes like in data-analyses) “portfolio/**/**/*.ipynb”
# https://stackoverflow.com/questions/62753648/rpc-failed-http-400-curl-22-the-requested-url-returned-error-400-bad-request
# git config --global http.postBuffer 524288000
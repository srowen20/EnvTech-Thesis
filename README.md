# Environmental Technology Masters Thesis Code
**Project Title:** Advancing Remote Sensing and Machine Learning Approaches for Soil Organic Carbon and Nitrogen Monitoring in Agriculture.

This repository contains the code for Sentinel-2, Sentinel-1, Landsat-8, Open-Meteo and LUCAS 2018 data collection, preprocessing, modelling and evaluation. 


## Collect Satellite Data from Google Earth Engine
This section details how to run the scripts to download data from Sentinel-2, Sentinel-1, and Landsat-8, from Google Earth Engine. The files required to collect this data are:
* `./data_collection/landsat_bands.py`
* `./data_collection/sentinel2_bands_and_cloud.py`
* `./data_collection/sentinel1_bands.py`

### Set-up
To run the any data collection from Google Earth Engine, first authenticate with Google Earth Engine by running the following in your Python environment:
```
earthengine authenticate
```
This step ensures access to the Earth Engine API for remote sampling.

### Data Extraction
The script must be run from the command line. 

Use the following command format to execute the script:
```
python sentinel_collect_bands_and_cloud.py --df_path <path/to/your/csv> --save_path <output_path.txt> [additional options]
```

For Example:
```
python data_collection/sentinel2_bands_and_cloud.py \
    --df_path data/modelling/baseline_datasets/processed/for_sentinel2_biggerwindow_leastcloudy_collection/20250802_leastcloudy_imagedateassurveydate.csv \
    --save_path ./data/raw/20250802_sentinel_bands_and_cloud_0-32days_radius5.txt \
    --num_workers 8 \
    --start_date_offset 32 \
    --end_date_offset 1 \
    --radius 5
```
Replace `sentinel2_bands_and_cloud.py` with the relevant filename. 

Adjust arguments as needed:
* `--start_date_offset <int>`: How many days before each soil survey date to start the image search.
* `--end_date_offset <int>`: How many days after each soil survey date to end the image search. If set to a negative number, e.g. -6, it will end with that many days before the soil survey date, e.g. 6 days before. 
* `--stratified True/False`: Stratified sampling by country, if not collecting all the sample satellite images.
* `--num_samples <int>`: Number of points to sample
* `--done_list True/False` and `--done_path <path>`: Resume from previously processed points.
* `--radius <int>`: the window radius. Setting to 1 means a 1 pixel border will be added around the centre pixel, 2 means a 5x5 kernel, 3 a 7x7 kernel etc.

### Output
All scripts output a .txt file containing line-delimited JSON records for each sampled point, including Sentinel-2 band values and cloud probability metrics.

### Notes
The script processes points in parallel, using the specified number of workers for speed.

If resuming a run, use the same `save_path`, and the programme will automatically detect the LUCAS survey POINT IDs which have already been completed. 

Customise sampling window and radius to fit your survey period and spatial requirements.


## Weather Data Collection Script (`openmeteo_weather.py`)

This script automates the collection of historical weather data from the [Open-Meteo API](https://open-meteo.com/), joining **daily and hourly variables** for each soil survey point in a CSV file. It is designed to work with the LUCAS 2018 soil dataset (or any dataset containing `POINTID`, `gps_lat`, `gps_long`, `SURVEY_DATE`, `Elev`).  

The script:  
- Reads a cleaned soil survey dataset (`--df_path`).  
- Creates a date window around each soil survey date (`--start_date_offset`, `--end_date_offset`).  
- Queries the Open-Meteo API for daily and hourly weather variables.  
- Aggregates hourly data (mean, min, max) and merges with daily values.  
- Saves results as line-delimited JSON records to a text file (`--save_path`).  
- Supports parallel processing (`--num_workers`) and skips already-processed points if provided with a list (`--done_list`, `--done_path`).  

### Usage

Run the script from the command line:

```bash
python openmeteo_weather.py \
  --df_path ./data/cleaned_data/250627_lucas_2018_cleaned.csv \
  --start_date_offset 31 \
  --end_date_offset 0 \
  --save_path ./data/raw/openmeteo_weather.txt \
  --num_workers 3

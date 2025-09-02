# Environmental Technology Masters Thesis Code
**Project Title:** Advancing Remote Sensing and Machine Learning Approaches for Soil Organic Carbon and Nitrogen Monitoring in Agriculture.
**Author:** Sophie Owen

## Repository Overview

This repository provides a complete pipeline for preparing, processing, and modelling soil monitoring data using remote sensing (RS) and machine learning (ML).  
It combines the **LUCAS 2018 topsoil dataset**, satellite imagery (Sentinel-1, Sentinel-2, Landsat-8), and weather data (Open-Meteo) to predict Soil Organic Carbon (SOC) and Nitrogen (N).

---

## Folder Structure

### `data/`
Contains input and output data files, including cleaned soil datasets, intermediate features, and raw extracted band/weather files.  
This is where all scripts and notebooks read from and write to.

### `data_collection/`
Python scripts for large-scale data extraction and preprocessing. These can be run directly from the command line.

- **landsat_bands.py**  
  Collects **Landsat-8** surface reflectance and auxiliary bands (scaled appropriately).  
  Samples values around each LUCAS soil survey point over a user-specified temporal window.  
  Saves results as line-delimited JSON.

- **sentinel2_bands_and_cloud.py**  
  Collects **Sentinel-2** spectral bands (10m, 20m, 60m resolution) and links them with **Sentinel-2 Cloud Probability** products.  
  Extracts both center pixel and small-patch neighbourhood values, as well as cloud probabilities.  

- **sentinel1_bands.py**  
  Collects **Sentinel-1** radar backscatter features (VV, VH).  
  Extracts both center pixel values and neighbourhood patches (e.g. 3×3).  

- **openmeteo_weather.py**  
  Automates historical weather data collection from the **Open-Meteo API**, joining daily and hourly variables for each soil survey point.  
  Aggregates hourly variables (mean/min/max) and merges with daily data.  

Each of these scripts accepts command-line arguments such as `--df_path` (input CSV of soil points), `--start_date_offset`, `--end_date_offset`, `--save_path`, and `--num_workers`.  
See the individual script docstrings/`--help` flag for details.

### `utils/`
Utility classes and helper functions.

- **ModelTrainer.py**  
  Defines a `ModelTrainer` class that streamlines model training, hyperparameter tuning, evaluation, and feature importance logging.  
  Supports **XGBoost, LightGBM, CatBoost, Random Forest, and SVR**.  
  Includes Optuna-based tuning, SHAP feature importances, RFE, permutation importance, and Excel logging of results.  

### Root-level Notebooks
Jupyter notebooks that explain and demonstrate each step of the pipeline. These are exploratory and analytical, and should be run in order:

1. **00_lucas_cleaning.ipynb** — Cleans and filters the raw LUCAS 2018 soil dataset.  
2. **01_lucas_exploration.ipynb** — Exploratory analysis of cleaned soil data (distributions, correlations, sampling biases).  
3. **02a_landsat_preprocessing.ipynb** — Preprocesses Landsat-8 imagery (cloud filtering, temporal alignment).  
4. **02b_sentinel_2_preprocessing.ipynb** — Preprocesses Sentinel-2 imagery (spectral bands, red-edge, SWIR features).  
5. **02c_sentinel_1_preprocessing.ipynb** — Preprocesses Sentinel-1 radar imagery.  
6. **02d_openmeteo_preprocessing.ipynb** — Processes Open-Meteo weather data and aligns it with soil survey dates.  
7. **03_joining_datasets.ipynb** — Joins soil, satellite, and weather datasets into a unified feature table.  
8. **04_modelling.ipynb** — Trains and evaluates ML models (using `ModelTrainer`) for SOC and Nitrogen prediction. Includes feature importance analysis and experimental comparisons.  

---

## Running the Pipeline

1. **Prepare the soil dataset**  
   Run `00_lucas_cleaning.ipynb` to clean the raw LUCAS data.  

2. **Collect remote sensing and weather features**  
   Use the scripts in `data_collection/` (e.g. `landsat_bands.py`, `sentinel2_bands_and_cloud.py`) to collect features for each soil point. Each script saves outputs into `./data/raw/`.  

3. **Preprocess and join data**  
   Use notebooks `02a`–`03` to preprocess each dataset and join them into a single modelling table.  

4. **Run models**  
   Train and evaluate models using `04_modelling.ipynb`. The `ModelTrainer` class in `utils/ModelTrainer.py` provides reusable training pipelines.  

---

This modular structure allows you to re-run individual components (e.g. re-collect Sentinel-2 bands with a new time window, or test different ML models) without rebuilding the whole workflow.

The following sections explain how to run the `.py` scripts. 

## Satellite Data Collection Scripts

The `data_collection/` folder contains three scripts for extracting satellite features for each soil survey point:

- `sentinel2_bands_and_cloud.py`  
- `landsat_bands.py`  
- `sentinel1_bands.py`  

Each script connects to **Google Earth Engine (GEE)**, samples imagery around soil survey points, and saves the results as line-delimited JSON files in the `data/` folder.  

### Common Features

All three scripts share a set of common design choices that make large-scale data collection easier and more reliable:

- **Checkpointing / Resume support**  
  Each run checks the `--save_path` file and automatically skips point IDs that have already been processed. This allows you to stop and restart data collection at any time without duplicating work.  

- **Parallel processing**  
  Sampling is distributed across multiple workers (`--num_workers`), with a progress bar provided by `tqdm`.  

- **Flexible temporal windows**  
  Date ranges are set relative to the soil survey date with `--start_date_offset` and `--end_date_offset`. For example, `--start_date_offset 62 --end_date_offset 0` collects imagery up to 62 days before the survey date.  

- **Optional stratified sampling**  
  If `--stratified True` is passed, the script performs stratified sampling by country, ensuring a representative subset of points. The number of samples can be controlled with `--num_samples`.  

- **Neighbourhood patch extraction**  
  Instead of only extracting the center pixel, each script also collects values from a neighbourhood (using `--radius`), creating patch-level arrays (e.g. 3×3 or 5×5).  

### Sentinel-2 (`sentinel2_bands_and_cloud.py`)

- **Data source:** `COPERNICUS/S2_SR_HARMONIZED` (surface reflectance) + `COPERNICUS/S2_CLOUD_PROBABILITY` (cloud mask).  
- **Bands:** 10 m (B2, B3, B4, B8, AOT, WVP), 20 m (B5, B6, B7, B8A, B11, B12, SCL), 60 m (B1, B9).  
- **Special features:**  
  - Extracts both center pixel values and patch arrays for each band.  
  - Samples cloud probability at the center pixel and as a patch array.  
  - Outputs include both spectral features and cloud metrics.  

### Landsat-8 (`landsat_bands.py`)

- **Data source:** `LANDSAT/LC08/C02/T1_L2` (surface reflectance + atmospheric/thermal bands).  
- **Bands:** Standard Landsat-8 bands, including useful atmospheric indices.  
- **Special features:**  
  - Includes atmospheric bands such as emissivity (important for SOC/N modelling).  
  - Same patch + center pixel extraction as Sentinel-2.  

### Sentinel-1 (`sentinel1_bands.py`)

- **Data source:** `COPERNICUS/S1_GRD`.  
- **Bands:** Radar backscatter (`VV`, `VH`).  
- **Special features:**  
  - Supports both center pixel and patch extraction.  
  - Handles both ascending and descending passes (configurable if needed).  
  - Can be extended to use different orbit or polarisation filters.  


### Example Command

Each script can be run directly from the command line. For example, to extract Sentinel-2 features:

```
python data_collection/sentinel2_bands_and_cloud.py \
  --df_path ./data/processed/20250711_lucas_cleaned.csv \
  --start_date_offset 62 \
  --end_date_offset 0 \
  --save_path ./data/raw/sentinel2_features.txt \
  --num_workers 6 \
  --radius 5 \
  --stratified False
```

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


## Weather Data Collection Script (`./data_collection/openmeteo_weather.py`)

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

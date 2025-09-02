import os
import json
import argparse
from pandas.tseries.offsets import DateOffset
from time import sleep
import time

from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta


# Get completed point ids
def get_completed_ids(save_path: str,
                      done_list: bool=False,
                      done_path: str=None) -> set:
    """
    Reads a line-delimited JSON file, containing Point IDs which have been processed already.

    Args:
        save_path (str): Path to the text file containing saved result records (one JSON per line).

    Returns:
        completed_ids (set): A set of 'point_id' values that have already been processed. 
    """
    if done_list == True:
        print("Extracting list of completed IDs.")
        with open(done_path, 'r') as f:
            completed_ids = [int(line.rstrip()) for line in f]
        print(f"{len(completed_ids)} IDs complete.")
    else:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # If the file doesn't exist yet, return an empty set
        if not os.path.exists(save_path):
            print("No IDs currently saved.")
            return set()
        
        completed_ids = set()
        with open(save_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    completed_ids.add(record['point_id'])
                except:
                    # Skip lines that are not valid JSON format
                    continue


    return completed_ids


def process_df(filepath: str, 
                      start_date_offset: int, 
                      end_date_offset: int, 
                      completed_ids: list
                      ) -> pd.DataFrame:
    """
    Loads a CSV of soil survey points, filters out completed ones, and adds
    Earth Engine-compatible date ranges for each row. As an additional option, 
    applies stratified sampling by country.

    Args: 
        filepath (str):
    """
    
    # Read the csv file
    df = pd.read_csv(filepath)

    # Filter out completed points
    df = df[~df['POINTID'].isin(completed_ids)]

    # Create start and end_date_cols
    df['survey_date'] = df['SURVEY_DATE'].copy().apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    df['start_date'] = df['survey_date'].copy().apply(lambda x: (x - timedelta(days=start_date_offset)).strftime("%Y-%m-%d"))

    if end_date_offset==0:
        df['end_date'] = df['survey_date'].copy().apply(lambda x: x + DateOffset(days=1))
    else:
        df['end_date'] = df['survey_date'].copy().apply(lambda x: x + DateOffset(days=end_date_offset+1))

    # Create a stratified sample by country name 
    print(f"Processing {len(df)} points from {start_date_offset} day(s) before to {end_date_offset} day(s) after survey date...")

    return df


def safe_weather_api_call(openmeteo, url, params, start_time=None):
    try:
        return openmeteo.weather_api(url, params=params)
    except Exception as e:
        err_msg = str(e)
        if 'Minutely API request limit exceeded' in err_msg:
            # If a start_time is passed, use it to determine how long to sleep
            if start_time:
                elapsed = time.time() - start_time
                to_sleep = max(60 - elapsed, 1)  # Ensure we sleep at least 1 second
                print(f"[Rate limit] Elapsed: {elapsed:.2f}s. Sleeping for {to_sleep:.2f}s...")
                time.sleep(to_sleep)
            else:
                print("[Rate limit] No start time passed. Sleeping for 60s as fallback.")
                time.sleep(60)
            return safe_weather_api_call(openmeteo, url, params, start_time=time.time())  # restart timer
        else:
            raise


def process_row(row, save_path, lock, url, daily_params, hourly_params):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
		"latitude": row['gps_lat'],
		"longitude": row['gps_long'],
		"elevation": row["Elev"],
		"start_date": row['start_date'],
		"end_date": row['end_date'],
		"daily": ",".join(daily_params),
		"timezone": "auto",
		"hourly": ",".join(hourly_params)
	}
    
    # responses = openmeteo.weather_api(url, params=params)
    start_time = time.time()

    responses = safe_weather_api_call(openmeteo, url, params, start_time=start_time)
    response = responses[0]

    point_id = row['POINTID']
    survey_date = row['survey_date']


    def process_response(data, param_list, time_label):
        time_index = pd.date_range(
            start = pd.to_datetime(data.Time(), unit = "s", utc = True),
            end = pd.to_datetime(data.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = data.Interval()),
            inclusive = "left"
        )

        values = {
            param: data.Variables(i).ValuesAsNumpy() for i, param in enumerate(param_list)
        }

        output_df = pd.DataFrame(values)
        output_df[time_label] = time_index
        output_df['point_id'] = point_id
        output_df['survey_date'] = survey_date
        
        return output_df


    hourly_df = process_response(response.Hourly(), hourly_params, "datetime")
    daily_df = process_response(response.Daily(), daily_params, "date")

    # Aggregate hourly parameters
    hourly_df['date'] = hourly_df['datetime'].dt.date
    agg_funcs = ['mean', 'min', 'max']
    hourly_agg = (
        hourly_df
        .groupby(["point_id", "survey_date", "date"])[hourly_params]
        .agg(agg_funcs)
    )

    # Flatten columns
    hourly_agg.columns = hourly_agg.columns.map('_'.join)
    hourly_agg = hourly_agg.reset_index()

    # Format daily for merge
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    daily_df = daily_df.drop(columns=["datetime"], errors="ignore")
    # Merge and return
    merged_df = pd.merge(daily_df, hourly_agg, on=["point_id", "survey_date", "date"], how="outer")
    merged_list = merged_df.to_dict(orient="records")

    if merged_list is None:
        print("Nothing being Returned")
        return

    with lock:
        with open(save_path, 'a') as f:
            for row_dict in merged_list:
                f.write(json.dumps(row_dict, default=str) + '\n')


def starmap_wrapper(args):
    """
    imap_unordered takes a singular argument. This function enables the use of multi-arg functions.
    
    """
    row, save_path, lock, url, daily_params, hourly_params = args
    return process_row(row, save_path, lock, url, daily_params, hourly_params)


def run_parallel_sampling(save_path, df, num_workers):
    manager = Manager()
    lock = manager.Lock()

    required_cols = ['gps_long', 'gps_lat', 'start_date', 'end_date', 'POINTID', 'survey_date', 'Elev']
    row_dicts = df[required_cols].copy()

    row_dicts['start_date'] = row_dicts['start_date'].astype(str)
    row_dicts['end_date'] = row_dicts['end_date'].astype(str)
    row_dicts['survey_date'] = row_dicts['survey_date'].astype(str)
    row_dicts = row_dicts.to_dict('records')

    


    daily_params = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum", "sunrise", "sunshine_duration", "daylight_duration", "wind_speed_10m_max", "wind_gusts_10m_max", "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
    hourly_params = ["relative_humidity_2m", "cloud_cover", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"]
    url = "https://archive-api.open-meteo.com/v1/archive"

    args_list = [(row, save_path, lock, url, daily_params, hourly_params) for row in row_dicts]
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(row_dicts), desc="Processing points") as pbar:
            for _ in pool.imap_unordered(starmap_wrapper, args_list):
                pbar.update()

def main():

    parser = argparse.ArgumentParser(description="Run weather API collection pipeline.")
    parser.add_argument("--df_path", type=str, default="./data/cleaned_data/250627_lucas_2018_cleaned.csv", help="Soil file path.")
    parser.add_argument("--start_date_offset", type=int, default=33, help="Number of days before survey date to include.")
    parser.add_argument("--end_date_offset", type=int, default=0, help="Number of days after survey date to include.")
    parser.add_argument("--save_path", type=str, default="./data/raw/20250630_openmeteo_weather.txt", help="Txt file to save the sample outputs to.")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of workers for parallel processing.")
    parser.add_argument("--done_list", type=bool, default=False, help="True or False if a text file containing completed Point IDs is supplied.")
    parser.add_argument("--done_path", type=str, default="", help="File path to text file of completed IDs.")
    args = parser.parse_args()

    completed_ids = get_completed_ids(args.save_path, args.done_list, args.done_path)
    print(f"Skipping {len(completed_ids)} already processed points...")

    df = process_df(filepath=args.df_path, 
                        start_date_offset=args.start_date_offset, 
                        end_date_offset=args.end_date_offset, 
                        completed_ids=completed_ids,
                        )

    run_parallel_sampling(save_path=args.save_path,
                          df=df,
                          num_workers=args.num_workers,
                          )

if __name__ == "__main__":
    main()
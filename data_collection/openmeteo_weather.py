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


def get_completed_ids(save_path: str,
                      done_list: bool = False,
                      done_path: str = None) -> set:
    '''
    Returns the set of POINTID values already processed so runs can resume safely.

    Parameters:
        save_path (str): Path to the line-delimited JSON output file (one JSON per line).
        done_list (bool): If True, read a plain-text list of completed IDs from done_path.
        done_path (str): Path to a text file with one POINTID per line (used if done_list=True).

    Returns:
        set: Set of integer POINTID values already processed.
    '''
    if done_list is True:
        print("Extracting list of completed IDs.")
        with open(done_path, 'r') as f:
            completed_ids = {int(line.rstrip()) for line in f if line.strip()}
        print(f"{len(completed_ids)} IDs complete.")
        return completed_ids

    # Ensure output directory exists so later appends won't fail
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # If no output yet, nothing has been processed
    if not os.path.exists(save_path):
        print("No IDs currently saved.")
        return set()

    completed_ids = set()
    with open(save_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                completed_ids.add(record['point_id'])
            except Exception:
                # Skip malformed lines to keep the run resilient
                continue
    return completed_ids


def process_df(filepath: str,
               start_date_offset: int,
               end_date_offset: int,
               completed_ids: set) -> pd.DataFrame:
    '''
    Loads the soil survey CSV, removes already-processed POINTIDs, and adds
    API-ready date windows (start_date/end_date) per row.

    Parameters:
        filepath (str): Path to the soil CSV. Requires POINTID, gps_lat, gps_long, SURVEY_DATE, Elev.
        start_date_offset (int): Days before SURVEY_DATE to include (e.g., 31).
        end_date_offset (int): Days after SURVEY_DATE to include (0 caps at the survey day+1).
        completed_ids (set): Set of POINTIDs already processed.

    Returns:
        pd.DataFrame: Filtered dataframe with 'survey_date', 'start_date', and 'end_date' columns added.
    '''
    df = pd.read_csv(filepath)

    # Sanity check for required columns (helps fail early with a clear error)
    required = {'POINTID', 'gps_lat', 'gps_long', 'SURVEY_DATE', 'Elev'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {filepath}: {missing}")

    # Skip completed points
    if completed_ids:
        df = df[~df['POINTID'].isin(completed_ids)]

    # Build window columns for the API (strings for Open-Meteo)
    df['survey_date'] = df['SURVEY_DATE'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    df['start_date'] = df['survey_date'].apply(lambda x: (x - timedelta(days=start_date_offset)).strftime("%Y-%m-%d"))

    # Open-Meteo end_date is exclusive when using hourly ranges; we add +1 day consistently
    if end_date_offset == 0:
        df['end_date'] = df['survey_date'].apply(lambda x: (x + DateOffset(days=1)).strftime("%Y-%m-%d"))
    else:
        df['end_date'] = df['survey_date'].apply(lambda x: (x + DateOffset(days=end_date_offset + 1)).strftime("%Y-%m-%d"))

    print(f"Processing {len(df)} points from {start_date_offset} day(s) before to {end_date_offset} day(s) after survey date...")
    return df


def safe_weather_api_call(openmeteo, url, params, start_time=None):
    '''
    Wraps the Open-Meteo API call with basic rate-limit handling and retry.

    Parameters:
        openmeteo (openmeteo_requests.Client): Configured Open-Meteo client.
        url (str): API endpoint URL.
        params (dict): Query parameters for the API.
        start_time (float): Epoch seconds when the call started (for adaptive sleep).

    Returns:
        list: List of Open-Meteo responses (first element is used).
    '''
    try:
        return openmeteo.weather_api(url, params=params)
    except Exception as e:
        err_msg = str(e)
        if 'Minutely API request limit exceeded' in err_msg:
            # Sleep to respect the per-minute limit, then retry
            if start_time:
                elapsed = time.time() - start_time
                to_sleep = max(60 - elapsed, 1)
                print(f"[Rate limit] Elapsed: {elapsed:.2f}s. Sleeping for {to_sleep:.2f}s...")
                time.sleep(to_sleep)
            else:
                print("[Rate limit] No start time passed. Sleeping for 60s as fallback.")
                time.sleep(60)
            return safe_weather_api_call(openmeteo, url, params, start_time=time.time())
        else:
            # Bubble up unknown errors
            raise


def process_row(row, save_path, lock, url, daily_params, hourly_params):
    '''
    Calls the Open-Meteo API for one survey point, aggregates hourly → daily,
    merges with daily variables, and appends JSON lines to the output.

    Parameters:
        row (dict): Row dict with gps_lat, gps_long, Elev, start_date, end_date, POINTID, survey_date.
        save_path (str): Path to the line-delimited JSON output file.
        lock (multiprocessing.synchronize.Lock): Inter-process lock for safe appends.
        url (str): Open-Meteo API base URL.
        daily_params (list): Daily variables to request.
        hourly_params (list): Hourly variables to request.

    Returns:
        None
    '''
    # Cached + retried session for resilience
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

    # Rate-limit aware request
    start_time = time.time()
    responses = safe_weather_api_call(openmeteo, url, params, start_time=start_time)
    response = responses[0]

    point_id = row['POINTID']
    survey_date = row['survey_date']

    def process_response(data, param_list, time_label):
        '''
        Converts an Open-Meteo time block (Hourly/Daily) to a tidy DataFrame.

        Parameters:
            data: Open-Meteo Hourly() or Daily() block.
            param_list (list): Variables requested for this block.
            time_label (str): Column name for the time dimension ('datetime' or 'date').

        Returns:
            pd.DataFrame: Tidy dataframe with variables, time column, point_id, survey_date.
        '''
        time_index = pd.date_range(
            start=pd.to_datetime(data.Time(), unit="s", utc=True),
            end=pd.to_datetime(data.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=data.Interval()),
            inclusive="left"
        )
        values = {param: data.Variables(i).ValuesAsNumpy() for i, param in enumerate(param_list)}
        out = pd.DataFrame(values)
        out[time_label] = time_index
        out['point_id'] = point_id
        out['survey_date'] = survey_date
        return out

    # Build hourly + daily frames
    hourly_df = process_response(response.Hourly(), hourly_params, "datetime")
    daily_df = process_response(response.Daily(), daily_params, "date")

    # Aggregate hourly → daily (mean/min/max per variable)
    hourly_df['date'] = hourly_df['datetime'].dt.date
    agg_funcs = ['mean', 'min', 'max']
    hourly_agg = (
        hourly_df
        .groupby(["point_id", "survey_date", "date"])[hourly_params]
        .agg(agg_funcs)
    )
    # Flatten MultiIndex columns like var_mean → 'var_mean'
    hourly_agg.columns = hourly_agg.columns.map('_'.join)
    hourly_agg = hourly_agg.reset_index()

    # Prepare daily for merge and combine
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    daily_df = daily_df.drop(columns=["datetime"], errors="ignore")
    merged_df = pd.merge(daily_df, hourly_agg, on=["point_id", "survey_date", "date"], how="outer")
    merged_list = merged_df.to_dict(orient="records")

    if not merged_list:
        print("Nothing being returned for this point.")
        return

    # Thread-safe append
    with lock:
        with open(save_path, 'a') as f:
            for row_dict in merged_list:
                f.write(json.dumps(row_dict, default=str) + '\n')


def starmap_wrapper(args):
    '''
    Enables passing multiple arguments through Pool.imap_unordered.

    Parameters:
        args (tuple): (row, save_path, lock, url, daily_params, hourly_params)

    Returns:
        None
    '''
    row, save_path, lock, url, daily_params, hourly_params = args
    return process_row(row, save_path, lock, url, daily_params, hourly_params)


def run_parallel_sampling(save_path: str, df: pd.DataFrame, num_workers: int) -> None:
    '''
    Dispatches parallel workers to call the API for each row and append results.

    Parameters:
        save_path (str): Output path for line-delimited JSON results.
        df (pd.DataFrame): Dataframe containing required columns for the API.
        num_workers (int): Number of parallel workers.

    Returns:
        None
    '''
    manager = Manager()
    lock = manager.Lock()

    # Only ship required columns to workers
    required_cols = ['gps_long', 'gps_lat', 'start_date', 'end_date', 'POINTID', 'survey_date', 'Elev']
    row_dicts = df[required_cols].copy()

    # Ensure JSON-serialisable
    row_dicts['start_date'] = row_dicts['start_date'].astype(str)
    row_dicts['end_date'] = row_dicts['end_date'].astype(str)
    row_dicts['survey_date'] = row_dicts['survey_date'].astype(str)
    row_dicts = row_dicts.to_dict('records')

    # Fixed variable sets and endpoint (change here if you need different variables)
    daily_params = [
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum",
        "sunrise", "sunshine_duration", "daylight_duration",
        "wind_speed_10m_max", "wind_gusts_10m_max",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration"
    ]
    hourly_params = [
        "relative_humidity_2m", "cloud_cover",
        "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
        "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"
    ]
    url = "https://archive-api.open-meteo.com/v1/archive"

    args_list = [(row, save_path, lock, url, daily_params, hourly_params) for row in row_dicts]
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(row_dicts), desc="Processing points") as pbar:
            for _ in pool.imap_unordered(starmap_wrapper, args_list):
                pbar.update()


def main():
    '''
    Parses CLI arguments, prepares the dataframe, and launches parallel collection.

    Parameters:
        None

    Returns:
        None
    '''
    parser = argparse.ArgumentParser(description="Run weather API collection pipeline.")
    parser.add_argument("--df_path", type=str, default="./data/cleaned_data/250627_lucas_2018_cleaned.csv",
                        help="Soil CSV path with POINTID, gps_lat, gps_long, SURVEY_DATE, Elev.")
    parser.add_argument("--start_date_offset", type=int, default=33,
                        help="Days before SURVEY_DATE to include.")
    parser.add_argument("--end_date_offset", type=int, default=0,
                        help="Days after SURVEY_DATE to include (0 caps at survey day).")
    parser.add_argument("--save_path", type=str, default="./data/raw/20250630_openmeteo_weather.txt",
                        help="Path to write line-delimited JSON results.")
    parser.add_argument("--num_workers", type=int, default=3,
                        help="Number of parallel workers.")
    parser.add_argument("--done_list", type=bool, default=False,
                        help="If True, read completed POINTIDs from --done_path.")
    parser.add_argument("--done_path", type=str, default="",
                        help="Path to text file with completed POINTIDs (one per line).")
    args = parser.parse_args()

    # Resume support: skip IDs we’ve already processed
    completed_ids = get_completed_ids(args.save_path, args.done_list, args.done_path)
    print(f"Skipping {len(completed_ids)} already processed points...")

    # Prepare dataframe with date windows
    df = process_df(filepath=args.df_path,
                    start_date_offset=args.start_date_offset,
                    end_date_offset=args.end_date_offset,
                    completed_ids=completed_ids)

    # Launch parallel API calls
    run_parallel_sampling(save_path=args.save_path,
                          df=df,
                          num_workers=args.num_workers)


if __name__ == "__main__":
    main()

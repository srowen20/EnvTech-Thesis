import pandas as pd
import ee
import argparse
from pandas.tseries.offsets import DateOffset
import json
import os

from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm

# Authenticate earth engine use and initialize
ee.Authenticate()
ee.Initialize(project='sophies-practice-project-1')


def get_completed_ids(save_path: str,
                      done_list: bool = False,
                      done_path: str = None) -> set:
    '''
    Returns the set of POINTID values already processed so runs can resume safely.

    Parameters:
        save_path (str): Path to the line-delimited JSON file where results are saved.
        done_list (bool): If True, read a plain-text list of completed IDs from done_path instead.
        done_path (str): Path to a text file containing one completed POINTID per line.

    Returns:
        set: A set of integer POINTID values already processed.
    '''
    if done_list is True:
        # Read a plain list of IDs from a text file
        print("Extracting list of completed IDs.")
        with open(done_path, 'r') as f:
            completed_ids = {int(line.rstrip()) for line in f if line.strip()}
        print(f"{len(completed_ids)} IDs complete.")
        return completed_ids

    # Ensure output directory exists (so we can create the file later if needed)
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
                # Expecting 'point_id' in each saved record
                completed_ids.add(record['point_id'])
            except Exception:
                # Skip any malformed lines
                continue
    return completed_ids


def process_df_for_ee(filepath: str,
                      start_date_offset: int,
                      end_date_offset: int,
                      completed_ids: set,
                      stratified: bool,
                      n: int = None) -> pd.DataFrame:
    '''
    Loads the soil points CSV, removes already-processed POINTIDs,
    and adds Earth Engine date-window columns for each row. Optionally
    performs stratified sampling by country.

    Parameters:
        filepath (str): Path to the cleaned soil CSV with POINTID, gps_lat, gps_long, SURVEY_DATE, country_name.
        start_date_offset (int): Days before SURVEY_DATE to include in the imagery window.
        end_date_offset (int): Days after SURVEY_DATE to include (use 0 to cap at SURVEY_DATE).
        completed_ids (set): Set of POINTIDs already processed (to skip).
        stratified (bool): If True, sample a fraction per country.
        n (int): Total number of rows to sample when stratified=True (approximate).

    Returns:
        pd.DataFrame: Filtered dataframe with added 'start_date' and 'end_date' columns.
    '''
    # Read the csv file
    df = pd.read_csv(filepath)

    # Basic column checks (helps fail fast with a clear message)
    required_cols = {'POINTID', 'gps_lat', 'gps_long', 'SURVEY_DATE'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {filepath}: {missing}")

    # Filter out completed points
    if completed_ids:
        df = df[~df['POINTID'].isin(completed_ids)]

    # Normalize survey date and build the window
    df['SURVEY_DATE'] = pd.to_datetime(df['SURVEY_DATE'])
    df['start_date'] = df['SURVEY_DATE'] - DateOffset(days=start_date_offset)
    if end_date_offset == 0:
        df['end_date'] = df['SURVEY_DATE']
    else:
        df['end_date'] = df['SURVEY_DATE'] + DateOffset(days=end_date_offset)

    # Optional: stratified sampling by country_name (if present)
    if stratified:
        if 'country_name' not in df.columns:
            raise ValueError("Stratified sampling requested but 'country_name' column is missing.")
        # Choose a fraction that roughly yields n rows overall; clamp to (0,1].
        if n is None or n <= 0:
            n = min(1000, len(df))  # sensible default if not provided
        frac = min(1.0, max(n / max(len(df), 1), 1.0 / max(len(df), 1)))
        # Group-wise sampling to preserve representation by country
        df = df.groupby('country_name', group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=42) if len(x) > 0 else x
        )

    print(f"Processing {len(df)} points from {start_date_offset} day(s) before to {end_date_offset} day(s) after survey date...")
    return df


def extract_samples_ee(point: ee.Geometry,
                       start_date: pd.Timestamp,
                       end_date: pd.Timestamp,
                       point_id: str,
                       survey_date: pd.Timestamp,
                       radius: int) -> list:
    '''
    Queries Landsat-8 L2 imagery in the date range and samples the center pixel
    (scale=30 m) for a set of reflectance and auxiliary bands for each image.

    Parameters:
        point (ee.Geometry): Earth Engine point geometry (lon, lat).
        start_date (pd.Timestamp): Window start date for imagery filtering.
        end_date (pd.Timestamp): Window end date for imagery filtering.
        point_id (str): Unique POINTID for the survey location.
        survey_date (pd.Timestamp): The original survey date (metadata only).
        radius (int): Pixel radius for neighbourhood stats (currently unused in this function).

    Returns:
        list: A list of dictionaries, one per image, with sampled band values and metadata.
    '''
    # Landsat-8 L2 surface reflectance collection
    ls = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(point)
        # .distinct('system:time_start')  # consider enabling if you see duplicates
    )

    ls_size = ls.size().getInfo()
    if ls_size == 0:
        # No imagery in window; return a single placeholder record
        print("Not collecting images")
        return [{
            'point_id': point_id,
            'survey_date': str(survey_date),
            'image_id': None,
            'image_date': None
        }]

    image_list = ls.toList(ls_size)
    imgs = [ee.Image(image_list.get(i)) for i in range(ls_size)]

    # Surface reflectance and useful auxiliary bands
    reflectance_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    extra_bands = ['ST_B10', 'QA_PIXEL', 'SR_ATMOS_OPACITY']
    band_names = reflectance_bands + extra_bands

    def sample_img(img) -> dict:
        try:
            # Apply Landsat L2 scaling to SR and ST bands
            def apply_scaling(img_in):
                # Scale SR bands (0.0000275, -0.2)
                sr_bands = [b for b in band_names if b.startswith("SR_B")]
                if sr_bands:
                    scaled_sr = img_in.select(sr_bands).multiply(0.0000275).add(-0.2)
                    img_in = img_in.addBands(scaled_sr, overwrite=True)

                # Scale ST_B10 (0.00341802, +149)
                if "ST_B10" in band_names:
                    scaled_st = img_in.select(["ST_B10"]).multiply(0.00341802).add(149.0)
                    img_in = img_in.addBands(scaled_st, overwrite=True)

                # QA_PIXEL and SR_ATMOS_OPACITY left as-is
                return img_in

            scaled_img = apply_scaling(img)

            # Sample center pixel (scale 30 m)
            sample = scaled_img.sample(point, scale=30).first()
            if sample is None:
                raise ValueError("Center sample is None")

            # Extract values for all bands present in the scaled image
            sample_dict = ee.Dictionary({b: sample.get(b) for b in scaled_img.bandNames().getInfo()})

            # Add basic image metadata (ID and acquisition date)
            img_meta = img.toDictionary(['system:index', 'system:time_start'])
            sample_dict = sample_dict.set('image_id', img_meta.get('system:index'))
            sample_dict = sample_dict.set('image_date', img_meta.get('system:time_start'))

            # Convert EE dictionary to a Python dict and format the date
            props = sample_dict.getInfo()
            return {
                'point_id': point_id,
                'survey_date': str(survey_date),
                'image_id': props.get('image_id'),
                'image_date': pd.to_datetime(props.get('image_date'), unit='ms').strftime('%Y-%m-%d')
                if props.get('image_date') else None,
                **{k: props.get(k) for k in props if k not in ['image_id', 'image_date']}
            }

        except Exception as e:
            # If any sampling step fails, write a placeholder record to maintain progress
            print(f"Sampling failed for point_id={point_id} with error: {e}")
            return {
                'point_id': point_id,
                'survey_date': str(survey_date),
                'image_id': None,
                'image_date': None
            }

    # Sample every image in the window
    results = [sample_img(img) for img in imgs]
    return results


def process_row(row: dict, save_path: str, lock, radius: int) -> None:
    '''
    Processes a single row (survey point): samples imagery and appends
    the results to the output file in a thread-safe manner.

    Parameters:
        row (dict): A dict containing gps_long, gps_lat, start_date, end_date, POINTID, SURVEY_DATE.
        save_path (str): Path to the line-delimited JSON output file.
        lock (multiprocessing.synchronize.Lock): Inter-process write lock.
        radius (int): Pixel radius for neighbourhood stats (passed through; not used here).

    Returns:
        None
    '''
    # Build EE point geometry
    point = ee.Geometry.Point([row['gps_long'], row['gps_lat']])
    start_date = row['start_date']
    end_date = row['end_date']
    point_id = row['POINTID']
    survey_date = row['SURVEY_DATE']

    # Collect samples for this point across all images in the window
    output_list = extract_samples_ee(point, start_date, end_date, point_id, survey_date, radius)

    # Append each image-sample dict to the output file
    for result_dict in output_list:
        if not isinstance(result_dict, dict):
            continue
        with lock:
            with open(save_path, 'a') as f:
                f.write(json.dumps(result_dict, default=str))
                f.write('\n')


def starmap_wrapper(args):
    '''
    Enables passing multiple arguments through Pool.imap_unordered.

    Parameters:
        args (tuple): (row, save_path, lock, radius)

    Returns:
        None
    '''
    row, save_path, lock, radius = args
    return process_row(row, save_path, lock, radius)


def run_parallel_sampling(save_path: str, df: pd.DataFrame, num_workers: int, radius: int) -> None:
    '''
    Iterates over all rows and dispatches parallel workers to sample imagery,
    writing line-delimited JSON records to save_path.

    Parameters:
        save_path (str): Output path for JSONL results.
        df (pd.DataFrame): Dataframe containing required columns for sampling.
        num_workers (int): Number of parallel workers.
        radius (int): Pixel radius for neighbourhood stats (passed through; not used here).

    Returns:
        None
    '''
    manager = Manager()
    lock = manager.Lock()

    # Ensure we only ship what we need to workers (keeps memory lower)
    required_cols = ['gps_long', 'gps_lat', 'start_date', 'end_date', 'POINTID', 'SURVEY_DATE']
    row_dicts = df[required_cols].copy()
    # Convert datetimes to strings to make them JSON-serialisable
    row_dicts['start_date'] = row_dicts['start_date'].astype(str)
    row_dicts['end_date'] = row_dicts['end_date'].astype(str)
    row_dicts['SURVEY_DATE'] = row_dicts['SURVEY_DATE'].astype(str)
    row_dicts = row_dicts.to_dict('records')

    args_list = [(row, save_path, lock, radius) for row in row_dicts]
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(row_dicts), desc="Processing points") as pbar:
            for _ in pool.imap_unordered(starmap_wrapper, args_list):
                pbar.update()


def main():
    '''
    Parses CLI arguments, prepares the dataframe, and runs parallel sampling.

    Parameters:
        None

    Returns:
        None
    '''
    parser = argparse.ArgumentParser(description="Run Landsat-8 sampling pipeline.")
    parser.add_argument("--df_path", type=str,
                        default="./data/modelling/baseline_datasets/processed/20250711_lucas_cleaned.csv",
                        help="Soil CSV path containing POINTID, gps_lat, gps_long, SURVEY_DATE.")
    parser.add_argument("--stratified", type=bool, default=False,
                        help="Whether to run stratified sampling by country_name.")
    parser.add_argument("--num_samples", type=int, default=120,
                        help="Approximate total rows to sample when --stratified True.")
    parser.add_argument("--start_date_offset", type=int, default=365,
                        help="Days before SURVEY_DATE to include.")
    parser.add_argument("--end_date_offset", type=int, default=-32,
                        help="Days after SURVEY_DATE to include (0 caps at SURVEY_DATE).")
    parser.add_argument("--save_path", type=str,
                        default="./data/raw/20250803_landsat_bands_centre_32-365days.txt",
                        help="Path to write line-delimited JSON results.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers.")
    parser.add_argument("--radius", type=int, default=1,
                        help="Pixel radius for neighbourhood stats (not used in this file).")
    parser.add_argument("--done_list", type=bool, default=False,
                        help="If True, use --done_path as a list of completed POINTIDs.")
    parser.add_argument("--done_path", type=str, default="./data/raw/soil_ids_done_landsat.txt",
                        help="Path to a text file with completed POINTIDs (one per line).")
    args = parser.parse_args()

    # Determine which POINTIDs to skip so re-runs don't duplicate work
    completed_ids = get_completed_ids(args.save_path, args.done_list, args.done_path)
    print(f"Skipping {len(completed_ids)} already processed points...")

    # Prepare dataframe, filtering out completed IDs and adding date windows
    df = process_df_for_ee(filepath=args.df_path,
                           start_date_offset=args.start_date_offset,
                           end_date_offset=args.end_date_offset,
                           completed_ids=completed_ids,
                           stratified=args.stratified,
                           n=args.num_samples)

    # Launch parallel sampling
    run_parallel_sampling(save_path=args.save_path,
                          df=df,
                          num_workers=args.num_workers,
                          radius=args.radius)


if __name__ == "__main__":
    main()

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

    # Ensure the directory exists (future writes won't fail)
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
                # Skip malformed lines
                continue
    return completed_ids


def process_df_for_ee(filepath: str,
                      start_date_offset: int,
                      end_date_offset: int,
                      completed_ids: set,
                      stratified: bool,
                      n: int = None) -> pd.DataFrame:
    '''
    Loads the soil CSV, removes already-processed POINTIDs, and adds
    Earth Engine date-window columns for each row. Optionally stratifies by country.

    Parameters:
        filepath (str): Path to the cleaned soil CSV; requires POINTID, gps_lat, gps_long, SURVEY_DATE.
        start_date_offset (int): Days before SURVEY_DATE to include in the imagery window.
        end_date_offset (int): Days after SURVEY_DATE to include (0 caps at SURVEY_DATE).
        completed_ids (set): Set of POINTIDs already processed (to skip).
        stratified (bool): If True, sample a fraction per country_name.
        n (int): Approximate total rows to sample when stratified=True.

    Returns:
        pd.DataFrame: Filtered dataframe with 'start_date' and 'end_date' columns added.
    '''
    # Read the csv file
    df = pd.read_csv(filepath)

    # Basic column checks (fail fast with clear message)
    required_cols = {'POINTID', 'gps_lat', 'gps_long', 'SURVEY_DATE'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {filepath}: {missing}")

    # Filter out completed points
    if completed_ids:
        df = df[~df['POINTID'].isin(completed_ids)]

    # Create start and end_date columns
    df['SURVEY_DATE'] = pd.to_datetime(df['SURVEY_DATE'])
    df['start_date'] = df['SURVEY_DATE'] - DateOffset(days=start_date_offset)
    if end_date_offset == 0:
        df['end_date'] = df['SURVEY_DATE']
    else:
        df['end_date'] = df['SURVEY_DATE'] + DateOffset(days=end_date_offset)

    # Optional stratified sample by country_name
    if stratified:
        if 'country_name' not in df.columns:
            raise ValueError("Stratified sampling requested but 'country_name' column is missing.")
        if n is None or n <= 0:
            n = min(1000, len(df))  # sensible default if not provided
        frac = min(1.0, max(n / max(len(df), 1), 1.0 / max(len(df), 1)))
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
    Queries Sentinel-1 GRD within the date range and samples center-pixel values
    plus a neighbourhood array for VV and VH bands.

    Parameters:
        point (ee.Geometry): Earth Engine point geometry (lon, lat).
        start_date (pd.Timestamp): Imagery window start date.
        end_date (pd.Timestamp): Imagery window end date.
        point_id (str): Unique POINTID for the survey location.
        survey_date (pd.Timestamp): Original survey date (metadata only).
        radius (int): Pixel radius for neighbourhood arrays (e.g., 1=3x3, 2=5x5).

    Returns:
        list: List of dictionaries (one per image) with sampled values and metadata.
    '''
    s1_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start_date, end_date)
        .filterBounds(point)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        # .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))  # enable if needed
    )

    s1_size = s1_col.size().getInfo()
    if s1_size == 0:
        # No imagery in window; return a single placeholder record
        return [{
            'point_id': point_id,
            'survey_date': str(survey_date),
            'image_id': None,
            'image_date': None
        }]

    image_list = s1_col.toList(s1_size)
    imgs = [ee.Image(image_list.get(i)) for i in range(s1_size)]

    def sample_img(img) -> dict:
        try:
            # Define a square kernel for neighbourhood extraction
            kernel = ee.Kernel.square(radius=radius, units='pixels')
            bands = ['VV', 'VH']

            # Center pixel values at 10 m
            center_sample = img.select(bands).sample(point, scale=10).first()
            center_dict = ee.Dictionary({f"{b}_center": center_sample.get(b) for b in bands})

            # Patch (neighbourhood) arrays
            patch_img = ee.Image.cat([
                img.select([b]).neighborhoodToArray(kernel).rename(f"{b}_array") for b in bands
            ])
            patch_sample = patch_img.sample(point, scale=10).first()
            patch_dict = ee.Dictionary({f"{b}_array": patch_sample.get(f"{b}_array") for b in bands})

            # Merge dictionaries and add metadata
            full_dict = ee.Dictionary({}).combine(center_dict).combine(patch_dict)
            img_metadata = img.toDictionary(['system:index', 'system:time_start'])
            full_dict = full_dict.set('image_id', img_metadata.get('system:index'))
            full_dict = full_dict.set('image_date', img_metadata.get('system:time_start'))

            props = full_dict.getInfo()
            return {
                'point_id': point_id,
                'survey_date': str(survey_date),
                'image_id': props.get('image_id'),
                'image_date': pd.to_datetime(props.get('image_date'), unit='ms').strftime('%Y-%m-%d')
                if props.get('image_date') else None,
                **{k: props.get(k) for k in props if k not in ['image_id', 'image_date']}
            }

        except Exception as e:
            # Keep the pipeline moving on individual failures
            print(f"Sampling failed for point_id={point_id} with error: {e}")
            return {
                'point_id': point_id,
                'survey_date': str(survey_date),
                'image_id': None,
                'image_date': None,
            }

    # Sample every image in the window
    results = [sample_img(img) for img in imgs]
    return results


def process_row(row: dict, save_path: str, lock, radius: int) -> None:
    '''
    Processes a single survey point: samples imagery and appends the
    results to the output file in a thread-safe manner.

    Parameters:
        row (dict): Dict with gps_long, gps_lat, start_date, end_date, POINTID, SURVEY_DATE.
        save_path (str): Path to the line-delimited JSON output file.
        lock (multiprocessing.synchronize.Lock): Inter-process write lock.
        radius (int): Pixel radius for neighbourhood arrays.

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
        radius (int): Pixel radius for neighbourhood arrays.

    Returns:
        None
    '''
    manager = Manager()
    lock = manager.Lock()

    # Only pass the required columns to workers
    required_cols = ['gps_long', 'gps_lat', 'start_date', 'end_date', 'POINTID', 'SURVEY_DATE']
    row_dicts = df[required_cols].copy()
    # Convert datetimes to strings for JSON serialisation
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
    Parses CLI arguments, prepares the dataframe, and launches parallel sampling.

    Parameters:
        None

    Returns:
        None
    '''
    parser = argparse.ArgumentParser(description="Run Sentinel-1 sampling pipeline.")
    parser.add_argument("--df_path", type=str,
                        default="data/modelling/baseline_datasets/processed/20250711_lucas_cleaned.csv",
                        help="Soil CSV path.")
    parser.add_argument("--stratified", type=bool, default=False,
                        help="Run stratified sampling of the soil data - True or False.")
    parser.add_argument("--num_samples", type=int, default=120,
                        help="Number of soil samples to run (when stratified=True).")
    parser.add_argument("--start_date_offset", type=int, default=0,
                        help="Days before survey date to include.")
    parser.add_argument("--end_date_offset", type=int, default=1,
                        help="Days after survey date to include.")
    parser.add_argument("--save_path", type=str, default="./data/raw/20250720_sentinel_1_bands.txt",
                        help="Txt file to save the sample outputs to.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for parallel processing.")
    parser.add_argument("--radius", type=int, default=1,
                        help="Pixel radius for neighbourhood arrays (1=3x3, 2=5x5, ...).")
    parser.add_argument("--done_list", type=bool, default=False,
                        help="True or False if a text file containing completed Point IDs is supplied.")
    parser.add_argument("--done_path", type=str, default="",
                        help="File path to text file of completed IDs.")
    args = parser.parse_args()

    # Resume support: skip IDs we’ve already processed
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

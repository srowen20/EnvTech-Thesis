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


def process_df_for_ee(filepath: str, 
                      start_date_offset: int, 
                      end_date_offset: int, 
                      completed_ids: list, 
                      stratified: bool, 
                      n: int = None
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
    df['SURVEY_DATE'] = pd.to_datetime(df['SURVEY_DATE'])
    df['start_date'] = df['SURVEY_DATE'] - DateOffset(days=start_date_offset)

    if end_date_offset==0:
        df['end_date'] = df['SURVEY_DATE']
    else:
        df['end_date'] = df['SURVEY_DATE'] + DateOffset(days=end_date_offset)

    # Create a stratified sample by country name 
    if stratified:
        frac = max(n / len(df), 1 / len(df))
        df = df.groupby('country_name', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
    print(f"Processing {len(df)} points from {start_date_offset} day(s) before to {end_date_offset} day(s) after survey date...")

    return df


def extract_samples_ee(point: ee.Geometry, 
                       start_date: pd.Timestamp, 
                       end_date: pd.Timestamp, 
                       point_id: str, 
                       survey_date: pd.Timestamp,
                       radius
                       ) -> list:
    
    """
    Args:
        point (ee.Geometry): 
        start_date (pd.Timestamp):
        end_date (pd.Timestamp):
        point_id (str):
        survey_date (pd.Timestamp):
    """

    s1_col = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterDate(start_date, end_date) \
        .filterBounds(point) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        # .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

    s1_size = s1_col.size().getInfo()

    if s1_size == 0:
        return {
                'point_id': point_id,
                'survey_date': str(survey_date),
                'image_id': None,
                'image_date': None
            }

    image_list = s1_col.toList(s1_size)
    imgs = [ee.Image(image_list.get(i)) for i in range(s1_size)]


    def sample_img(img) -> dict:
        try:
            kernel = ee.Kernel.square(radius=radius, units='pixels')
            full_dict = ee.Dictionary({})

            bands = ['VV', 'VH']

            # Center pixel values
            center_sample = img.select(bands).sample(point, scale=10).first()
            center_dict = ee.Dictionary({f"{b}_center": center_sample.get(b) for b in bands})

            # 3x3 patch array values
            patch_img = ee.Image.cat([
                img.select([b]).neighborhoodToArray(kernel).rename(f"{b}_array") for b in bands
            ])
            patch_sample = patch_img.sample(point, scale=10).first()
            patch_dict = ee.Dictionary({f"{b}_array": patch_sample.get(f"{b}_array") for b in bands})

            # Combine all
            full_dict = full_dict.combine(center_dict).combine(patch_dict)

            img_metadata = img.toDictionary(['system:index', 'system:time_start'])
            full_dict = full_dict.set('image_id', img_metadata.get('system:index'))
            full_dict = full_dict.set('image_date', img_metadata.get('system:time_start'))

            props = full_dict.getInfo()

            return {
                'point_id': point_id,
                'survey_date': str(survey_date),
                'image_id': props.get('image_id'),
                'image_date': pd.to_datetime(props.get('image_date'), unit='ms').strftime('%Y-%m-%d') if props.get('image_date') else None,
                **{k: props.get(k) for k in props if k not in ['image_id', 'image_date']}
            }

        except Exception as e:
            print("Erroring here")
            return {
                'point_id': point_id,
                'survey_date': str(survey_date),
                'image_id': None,
                'image_date': None,
            }

    results = [sample_img(img) for img in imgs]

    return results


def process_row(row: dict, save_path: str, lock, radius):

    point = ee.Geometry.Point([row['gps_long'], row['gps_lat']])
    start_date = row['start_date']
    end_date = row['end_date']
    point_id = row['POINTID']
    survey_date = row['SURVEY_DATE']

    output_list = extract_samples_ee(point, start_date, end_date, point_id, survey_date, radius)

    for result_dict in output_list:
        if result_dict is None:
            continue

        if not isinstance(result_dict, dict):
            continue

        with lock:
            with open(save_path, 'a') as f:
                f.write(json.dumps(result_dict, default=str))
                f.write('\n')
            # print(f"Saved point ID: {result_dict['point_id']} - SCL: {result_dict['scl']}, Cloud Prob: {result_dict['cloud_probability']}")


def starmap_wrapper(args):
    """
    imap_unordered takes a singular argument. This function enables the use of multi-arg functions.
    
    """
    row, save_path, lock, radius = args
    return process_row(row, save_path, lock, radius)


def run_parallel_sampling(save_path, df, num_workers, radius):

    manager = Manager()
    lock = manager.Lock()
    
    required_cols = ['gps_long', 'gps_lat', 'start_date', 'end_date', 'POINTID', 'SURVEY_DATE']
    row_dicts = df[required_cols].copy()
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

    parser = argparse.ArgumentParser(description="Run cloud sampling pipeline.")
    parser.add_argument("--df_path", type=str, default="data/modelling/baseline_datasets/processed/20250711_lucas_cleaned.csv", help="Soil file path.")
    parser.add_argument("--stratified", type=bool, default=False, help="Run stratified sampling of the soil data - True or False.")
    parser.add_argument("--num_samples", type=int, default=120, help="Number of soil samples to run.")
    parser.add_argument("--start_date_offset", type=int, default=0, help="Number of days before survey date to include.")
    parser.add_argument("--end_date_offset", type=int, default=1, help="Number of days after survey date to include.")
    parser.add_argument("--save_path", type=str, default="./data/raw/20250720_sentinel_1_bands.txt", help="Txt file to save the sample outputs to.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for parallel processing.")
    parser.add_argument("--radius", type=int, default=1, help="Pixel radius for averaging values over.")
    parser.add_argument("--done_list", type=bool, default=False, help="True or False if a text file containing completed Point IDs is supplied.")
    parser.add_argument("--done_path", type=str, default="", help="File path to text file of completed IDs.")
    args = parser.parse_args()

    completed_ids = get_completed_ids(args.save_path, args.done_list, args.done_path)
    print(f"Skipping {len(completed_ids)} already processed points...")

    df = process_df_for_ee(filepath=args.df_path, 
                           start_date_offset=args.start_date_offset, 
                           end_date_offset=args.end_date_offset, 
                           completed_ids=completed_ids,
                           stratified=args.stratified, 
                           n=args.num_samples, 
                           )

    run_parallel_sampling(save_path=args.save_path,
                          df=df,
                          num_workers=args.num_workers,
                          radius=args.radius)



if __name__ == "__main__":
    main()
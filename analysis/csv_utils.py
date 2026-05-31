import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pandas import DataFrame

from train.log_callback import LogCallback


def load_history_from_csv(model_name: str) -> dict:
    df = pd.read_csv(f'{LogCallback.LOG_FOLDER}/{model_name}.csv')
    return df.to_dict(orient='list')


def load_dataframe(csv_path: str) -> DataFrame:
    df = pd.read_csv(csv_path)
    return df


def create_lesion_percentage_csv(leaf_masks_dir: str, spot_masks_dir: str):
    leaf_masks_dir = Path(leaf_masks_dir)
    spot_masks_dir = Path(spot_masks_dir)

    leaf_files = {p.stem: p for p in leaf_masks_dir.iterdir()}
    spot_files = {p.stem: p for p in spot_masks_dir.iterdir()}

    common_stems = sorted(set(set(leaf_files.keys()) & set(spot_files.keys())))
    output_csv = 'dataset/segment/spot/lesion_percentage.csv'

    with open(str(output_csv), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'lesion_percentage'])

        for stem in common_stems:
            # 'L' означает 8-bit grayscale
            leaf_mask = np.array(Image.open(leaf_files[stem]).convert('L'))
            spot_mask_orig = np.array(Image.open(spot_files[stem]).convert('L'))
            leaf_area = np.sum(leaf_mask > 0)
            lesion_area = np.sum(spot_mask_orig > 0)

            if leaf_area == 0:
                percentage = 0.0
            else:
                percentage = float(lesion_area) / float(leaf_area)
            writer.writerow([stem, percentage])


def calculate_extremums_in_column(df:DataFrame, column_name: str):
    col = df[column_name]

    max_val = col.max()
    min_val = col.min()
    mean_val = col.mean()

    count_max = (col == max_val).sum()

    print(f"Максимум: {max_val}")
    print(f"Минимум: {min_val}")
    print(f"Среднее: {mean_val}")
    print(f"Количество изображений с максимальным процентом: {count_max}")
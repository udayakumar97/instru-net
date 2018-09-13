"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
tqdm.monitor_interval = 0
data_path = Path('test')

train_path = data_path

cropped_train_path = Path('cropped_test')

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320


if __name__ == '__main__':
    for instrument_index in range(1, 11):
        instrument_folder = 'instrument_dataset_' + str(instrument_index)

        (cropped_train_path / instrument_folder / 'images').mkdir(exist_ok=True, parents=True)

        for file_name in tqdm(list((train_path / instrument_folder / 'left_frames').glob('*'))):
            img = cv2.imread(str(file_name))
            old_h, old_w, _ = img.shape

            img = img[h_start: h_start + height, w_start: w_start + width]
            cv2.imwrite(str(cropped_train_path / instrument_folder / 'images' / (file_name.stem + '.jpg')), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
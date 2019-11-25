import pytz
import datetime
import os
import numpy as np
from deepvoxels import data_util

def get_log_dir():
    # Result logging directory
    tz = pytz.timezone("US/Eastern")
    d = datetime.datetime.now(tz)
    dir_name = d.strftime('%y_%m_%d_%H_%M_%S')
    return dir_name

def load_rgb(path, img_size):
    # Borrowed from class NovelViewTriplets
    img = data_util.load_img(path, square_crop=True, downsampling_order=1, target_size=img_size)
    img = img[:, :, :3].astype(np.float32) / 255. - 0.5
    img = img.transpose(2,0,1)
    return img
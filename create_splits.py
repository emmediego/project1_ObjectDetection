import argparse
import glob
import os
import random

import numpy as np
import shutil

from utils import get_module_logger
from random import shuffle

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    dataset_files = glob.glob(data_dir + '/training_and_validation/*.tfrecord')
    shuffle(dataset_files)
    files_num = len(dataset_files)

    train_percent = .8
    train_end = int(train_percent * files_num)

    for src_path in dataset_files[:train_end]:
        dst_path = "{}/train/{}".format(data_dir, os.path.basename(src_path))
        shutil.move(src_path, dst_path)

    for src_path in dataset_files[train_end:]:
        dst_path = "{}/val/{}".format(data_dir, os.path.basename(src_path))
        shutil.move(src_path, dst_path)

    print(train_end, ' files moved to train folder')
    print(files_num - train_end, ' files moved to val folder')

    return

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
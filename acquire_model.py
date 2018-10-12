# -*- coding: utf-8 -*-
"""Downloads model-files (for retraining)."""

from helpers import io
import os
import random
import shutil
import zipfile
import wget
import time
import argparse



def download(url, dest_dir):
    print(f"\n Downloading Model from '{url}'..")

    if not os.path.exists(dest_dir): os.makedirs(dest_dir)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    tmp_filename = f"model-{timestr}.h5"
    dest_filepath = os.path.join(dest_dir, tmp_filename)
    wget.download(url, dest_filepath) 
    print("\nDownloaded", dest_filepath)

    return dest_filepath



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest_dir', type=str, nargs='?', default='./results/models')
    parser.add_argument('--model_url', type=str, required=True)
    args = parser.parse_args()

    model_path = download(args.model_url, args.dest_dir)
    print(model_path)

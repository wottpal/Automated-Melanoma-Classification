# -*- coding: utf-8 -*-
"""Downloads and/or unzips datasets."""

from helpers import io
import os
import random
import shutil
import zipfile
import wget
import time
import argparse



def download(url, tmp_path):
    """Downloads a .zip-file from the given `url` and saves it under the given `tmp_path`.
    
    Arguments:
        url {string} -- URL to the .zip-file (append `?dl=1` if it's a Dropbox-URL)
        tmp_path {string} -- Directory where the downloaded file should be saved
    
    Returns:
        string -- Path to the downloaded archive
    """
    print(f"\n Downloading Dataset from '{url}'..")

    if not os.path.exists(tmp_path): os.makedirs(tmp_path)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    tmp_filename = f"dataset-{timestr}.zip"
    tmp_filepath = os.path.join(tmp_path, tmp_filename)
    wget.download(url, tmp_filepath) 
    print("\nDownloaded", tmp_filepath)

    return tmp_filepath


def unzip_and_move(zip_filepath, data_path, clear_data_path=False):
    """Unzips the given file and moves all its child directories over to the `data_path`.
    
    Arguments:
        zip_filepath {string} -- Path to the .zip-file
        data_path {string} -- Path where the extraced directories should be copied to
    
    Keyword Arguments:
        clear_data_path {bool} -- If set to `True` the `data_path` will be cleared up front (default: {False})
    """

    # Clear data-directory
    if clear_data_path: io.clear_directories(data_path)

    # Unzip dataset
    extract_path = os.path.splitext(zip_filepath)[0] 
    zip_ref = zipfile.ZipFile(zip_filepath, 'r')
    zip_ref.extractall(extract_path)
    zip_ref.close()

    # Move all first-level dirs of the unzipped archive to the data-dir
    items = []
    items_root = ""
    for root, dirs, _ in os.walk(extract_path):
        if root == extract_path or "__MACOSX" in root: continue
        items = dirs
        items_root = root
        break

    for item in items:
        src = os.path.join(items_root, item)
        dest = os.path.join(data_path, item)
        shutil.move(src, dest)
        print("Created", dest)
    
    # Removed unzipped tmp-dir
    if os.path.isdir(extract_path): shutil.rmtree(extract_path)

    print("Unzipped & Moved", zip_filepath)


def extract_small_dataset(data_path, data_small_path, data_small_set_max):
    print(f"Create small extract of data-dir at '{data_small_path}'..")
    
    io.clear_directories(data_small_path)

    set_dirs = ['test', 'training', 'validation']
    for set_dir in set_dirs:
        path = os.path.join(data_path, set_dir)
        set_size = sum([len(files) for r, d, files in os.walk(path)])
        classes, images_per_class = io.subdirs_and_files(path)

        for idx, images in enumerate(images_per_class):
            new_size = int(len(images) / set_size * data_small_set_max)
            random.shuffle(images)
            new_images = images[:new_size]

            for image in new_images:
                src = os.path.join(data_path, set_dir, classes[idx], image)
                dest = os.path.join(data_small_path, set_dir, classes[idx], image)
                if not os.path.exists(os.path.dirname(dest)): os.makedirs(os.path.dirname(dest))
                shutil.copyfile(src, dest)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_path', type=str, nargs='?', default='./tmp')
    parser.add_argument('--data_path', type=str, nargs='?', default='./data')
    group_outer = parser.add_mutually_exclusive_group(required=True)
    group_outer.add_argument('--data_small_only', action='store_const', const=True)
    group = group_outer.add_mutually_exclusive_group()
    group.add_argument('--dataset_url', type=str)
    group.add_argument('--dataset_path', type=str)
    parser.add_argument('--data_small_path', type=str, nargs='?', default='./data_small')
    parser.add_argument('--data_small_set_max', type=int, default=150)
    args = parser.parse_args()

    if not args.data_small_only:
        dataset_url = args.dataset_url
        dataset_path = args.dataset_path or download(dataset_url, args.tmp_path)
        unzip_and_move(dataset_path, args.data_path, clear_data_path=True)

    extract_small_dataset(args.data_path, args.data_small_path, args.data_small_set_max)
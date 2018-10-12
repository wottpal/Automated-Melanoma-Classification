# -*- coding: utf-8 -*-
"""
Preparing available datasets for training, validation and evaluation by doing the following:
    - Sorting (transforming dataset-structure of available datasets i.e. ISIC into two folders 'benign' and 'malignant')
    - Resizing & Cropping
    - Balancing (duplicating images of underrepresented class esp. 'malignant')
    - Augmented Balancing (duplicate images of underrepresented class mostly with augmented versions)
    - Splitting (into the categories 'training', 'validation', 'test' with a given ratio)

ORIGINAL DATA PATHS:
    --src_path_isic "/Volumes/G-DRIVE/Datasets/AMC-Datasets/ISIC-Data (ORIGINAL)"
    --src_path_praxis "/Volumes/G-DRIVE/Datasets/AMC-Datasets/Hautpraxis-Data (ORIGINAL)" 
    --src_path_mednode "/Volumes/G-DRIVE/Datasets/AMC-Datasets/MEDNODE-Data (ORIGINAL)"
    --src_path_ph2 "/Volumes/G-DRIVE/Datasets/AMC-Datasets/PH2-Data (ORIGINAL)"

SAMPLE USAGE:
    - Sort ISIC: `python prepare_datasets.py --path "/Volumes/G-DRIVE/Datasets/AMC-Datasets/ISIC Sorted (Test)" --sort --src_path_isic "/Volumes/G-DRIVE/Datasets/AMC-Datasets/ISIC-Data (ORIGINAL)"`
    - Sort, Resize, Split, Balance ISIC: `python prepare_datasets.py --path "/Volumes/G-DRIVE/Datasets/AMC-Datasets/ISIC-Data Test (sor|res|spl|bal)" --sort --resize --split --balance --src_path_isic "/Volumes/G-DRIVE/Datasets/AMC-Datasets/ISIC-Data (ORIGINAL)"`
    - Balance Sorted Dataset: `python prepare_datasets.py --balance --path "/Volumes/G-DRIVE/Datasets/AMC-Datasets/Sample Dataset (sor|res|spl)"`
    - Sort, Resize & Split ALL DATA:
    ```
    python prepare_datasets.py --path "/Volumes/G-DRIVE/Datasets/AMC-Datasets/All Data (sor|res|spl)" --sort --resize --split --src_path_isic "/Volumes/G-DRIVE/Datasets/AMC-Datasets/ISIC-Data (ORIGINAL)" --src_path_praxis "/Volumes/G-DRIVE/Datasets/AMC-Datasets/Hautpraxis-Data (ORIGINAL)"  --src_path_mednode "/Volumes/G-DRIVE/Datasets/AMC-Datasets/MEDNODE-Data (ORIGINAL)" --src_path_ph2 "/Volumes/G-DRIVE/Datasets/AMC-Datasets/PH2-Data (ORIGINAL)"
    ```
"""

from sorter import isic, praxis, mednode, ph2
from preprocessors import resize, augmentation
from helpers import plotter, io
import os
import json
import cv2
import random
import math
import shutil
import uuid
import time
import numpy as np
from random import Random
from tqdm import tqdm
import argparse
import unicodedata
import sty 



def resize_images(path, size):
    print("Resize Images..")

    filepaths = []
    for root, _, files in os.walk(path):
        filepaths += [os.path.join(root, x) for x in files]

    for filepath in tqdm(filepaths):
        if not filepath.lower().endswith(('.jpg', '.jpeg')): continue
        image = cv2.imread(filepath)

        # Resize & Crop Image to `(size, size)`
        image = resize.apply(image, size)

        cv2.imwrite(filepath, image)


def sanitize_filenames(path):
    """To prevent e.g. German Umlauts in filenames etc."""
    filepaths = []
    for root, _, files in os.walk(path):
        filepaths += [os.path.join(root, x) for x in files]

    for filepath in tqdm(filepaths):
        path, filename = os.path.split(filepath)

        sanitized_filename = unicodedata.normalize('NFKD', filename).encode('ascii','ignore').decode('UTF-8')
        sanitized_filepath = os.path.join(path, sanitized_filename)

        if filename != sanitized_filename:
            os.rename(filepath, sanitized_filepath)
            print(filepath)
            print(f"-> {sanitized_filepath}\n")


def sort_datasets(args):
    """
    Sorts datasets with given path with their according sorter. 
    Additional arguments:
        - `args.sort_prefer_masked` (If True, a masked version of the image is used if available)
        - `args.sort_restrictive` (If True, uncertain diseases will be left out and only clear benign/malign images are copied)
    """

    if args.src_path_isic:
        print("Copy & Sort ISIC-Images..")
        isic.apply(args.src_path_isic, args.path, args.sort_restrictive, args.sort_prefer_masked)

    if args.src_path_praxis:
        print("Copy & Sort Hautpraxis-Images..")
        praxis.apply(args.src_path_praxis, args.path, args.sort_restrictive)

    if args.src_path_mednode:
        print("Copy & Sort MED-NODE-Images..")
        mednode.apply(args.src_path_mednode, args.path, args.sort_restrictive)

    if args.src_path_ph2:
        print("Copy & Sort PH2-Images..")
        ph2.apply(args.src_path_ph2, args.path, args.sort_restrictive, args.sort_prefer_masked)
    
    sanitize_filenames(args.path)


def balance(path, ratio = 1.0):
    """Balances multiple classes with the given ratio within (0,1] by duplicating images from the underrepresented class.
    NOTE: It assumes there are two classes only"""
    print(f"Balancing '{os.path.basename(path)}' classes to ratio {ratio:.2f}..")

    (classes, images) = io.subdirs_and_files(path, verbose=1)

    if len(classes) < 2:
        raise ValueError(f"Can't find two classes, check path. Found: {classes}")

    # Determine current class ratio
    len_0, len_1 = len(images[0]), len(images[1])
    currentRatio = round(min(len_0, len_1) / max(len_0, len_1), 3)

    class_cmp_str = f"'{classes[1]}' < '{classes[0]}'" if len_0 > len_1 else f"'{classes[0]}' < '{classes[1]}'"
    if len_0 == len_1: class_cmp_str = f"'{classes[0]}' = '{classes[1]}'"
    print(f"Current ratio is {currentRatio:.2f} ({class_cmp_str})")

    if currentRatio >= ratio: 
        print(f"Stopped Balancing, current ratio is already larger or equal than {ratio:.2f}")
        return
    
    len_create = int(ratio * max(len_0, len_1) - min(len_0, len_1))
    smallerClass = 0 if len_0 < len_1 else 1
    print(f"Duplicating {len_create} randomly chosen images from class '{classes[smallerClass]}'..")

    for i in range(len_create):
        rand_filename = random.choice(images[smallerClass])
        rand_path = os.path.join(path, classes[smallerClass], rand_filename)
        basename, extension, short_hash = os.path.splitext(rand_filename)[0], os.path.splitext(rand_filename)[1], str(uuid.uuid4())[:4]

        dupl_filename = f"{basename}-{short_hash}_DUPLICATE{extension}"
        dupl_path = os.path.join(path, classes[smallerClass], dupl_filename)
            
        shutil.copyfile(rand_path, dupl_path)


def augment(path, increase_factor):
    print("Augment Images..")

    if increase_factor <= 0.:
        raise ValueError(f"`increase_factor` must be larger than 0")
    
    filepaths = []
    for root, _, files in os.walk(path):
        filepaths += [os.path.join(root, x) for x in files]

    num_before, num_after = 0, 0
    for filepath in tqdm(filepaths):
        if not filepath.lower().endswith(('.jpg', '.jpeg')): continue

        num_before += 1
        image = cv2.imread(filepath)

        num_augmentations = int(increase_factor)
        num_augmentations += 1 if random.random() < (increase_factor - int(increase_factor)) else 0
        num_after += num_augmentations

        if num_augmentations <= 0: continue
        # print(f"{num_augmentations}x augment '{filepath}'..")

        for i in range(0, num_augmentations):
            augmented_image = augmentation.apply(image.copy())

            filename = os.path.basename(filepath)
            basename, extension, short_hash = os.path.splitext(filename)[0], os.path.splitext(filename)[1], str(uuid.uuid4())[:4]
            rand_new_filename = f"{basename}-{short_hash}_AUGMENTED_DUPLICATE{extension}"
            rand_new_path = os.path.join(path, rand_new_filename)

            # print(f"Writing {rand_new_path}..")
            cv2.imwrite(rand_new_path, augmented_image)

    num_after += num_before
    print(f"\n\nFactor={increase_factor}, Before={num_before}, After={num_after}\n\n")


def augment_and_balance(path, additional_increase_factor, balance_ratio = 1.0):
    """
    Augments larger class of the given set by factor `additional_increase_factor` (e.g. 1.0 means doubling all data).
    The under-representet class is even more augmented to match the `balance_ratio` with the other set (e.g. 1.0 means equal size).
    """
    print(f"Augmenting & Balancing '{os.path.basename(path)}' classes..")
    print(sty.ef.dim + f"[augmentation_increase_factor={additional_increase_factor}, balance_ratio={balance_ratio}]\n" + sty.rs.all)


    (classes, images) = io.subdirs_and_files(path, verbose=1)
    if len(classes) < 2:
        raise ValueError(f"Can't find two classes, check path. Found: {classes}")

    # Determine current class ratio
    len_0, len_1 = len(images[0]), len(images[1])
    currentRatio = round(min(len_0, len_1) / max(len_0, len_1), 3)

    class_cmp_str = f"'{classes[1]}' < '{classes[0]}'" if len_0 > len_1 else f"'{classes[0]}' < '{classes[1]}'"
    if len_0 == len_1: class_cmp_str = f"'{classes[0]}' = '{classes[1]}'"
    print(f"Current ratio is {currentRatio:.2f} ({class_cmp_str})")

    smallerClass, largerClass = 0 if len_0 < len_1 else 1, 1 if len_0 < len_1 else 0

    random.seed(42)
    if additional_increase_factor > 0: 
        augment(os.path.join(path, classes[largerClass]), additional_increase_factor)

    newRatio = len(images[smallerClass]) / (len(images[largerClass]) + additional_increase_factor * len(images[largerClass]))
    smaller_class_increase_factor = balance_ratio / newRatio - 1
    if smaller_class_increase_factor > 0:
        augment(os.path.join(path, classes[smallerClass]), smaller_class_increase_factor)
    


def revert_balance(path):
    """This deletes all images ending with '_DUPLICATE.{ext}'. 
    IMPORTANT: This is a desstructive operation, use it with caution."""
    print("Revert Balancing Dataset.. (IMPORTANT: Files are going to be deleted)")

    filepaths = []
    for root, _, files in os.walk(path):
        filepaths += [os.path.join(root, x) for x in files]

    for filepath in filepaths:
        _, filename = os.path.split(filepath)
        if not filename.lower().endswith(('_duplicate.jpg', '_duplicate.jpeg', '_duplicate.png')): continue
        print(f"Deleting {filepath}..")
        os.remove(filepath)



def split(path, test_split=.15, validation_split=.15, random=True):
    """iterative, so last (traning) must be one"""
    print("Split Images..")

    # Determine images for each class and shuffle them
    images = {}
    amounts = {}
    _, class_dirs, _ = list(os.walk(path))[0]
    for class_name in class_dirs:
        class_path = os.path.join(path, class_name)
        _, _, class_images = list(os.walk(class_path))[0]

        images[class_name] = class_images
        amounts[class_name] = len(class_images)
        if random: Random(42).shuffle(images[class_name])

    # Sort & copy images into each set with corresponding quota
    splits = { 
        'test': test_split, 
        'validation': validation_split, 
        'training': 1 - test_split - validation_split
    }

    for class_name, _ in images.items():
        # Determine actual split-amounts for each class
        split_amounts = { 
            'test': int(splits['test'] * amounts[class_name]), 
            'validation': int(splits['validation'] * amounts[class_name]), 
            'training': int(splits['training'] * amounts[class_name]), 
        }
        print(f"Split-Amounts for {class_name} ({amounts[class_name]}): {split_amounts}")

        for split_name, split_amount in split_amounts.items():
            split_class_dir = os.path.join(split_name, class_name)

            split_class_images = images[class_name][:split_amount]
            images[class_name] = images[class_name][split_amount:]

            dest_dir = os.path.join(path, split_class_dir)
            if not os.path.exists(dest_dir): os.makedirs(dest_dir)
            for image in split_class_images:
                src = os.path.join(path, class_name, image)
                dest = os.path.join(dest_dir, image)
                shutil.move(src, dest)
            
    # Remove old class-dirs
    for class_name in class_dirs:
        class_path = os.path.join(path, class_name)
        shutil.rmtree(class_path)
            

def revert_split(path):
    """IMPORTANT: Don't do this if already augmented, validation & test data should never be retouched."""
    print("Revert Split..")

    _, split_names, _ = list(os.walk(path))[0]
    _, class_names, _ = list(os.walk(path))[1]

    if not len(class_names): 
        raise ValueError("Can't revert split! Found no class-directories under split-directories")

    # Collect & copy images for each class
    for split_name in split_names:
        for class_name in class_names:
            split_class_path = os.path.join(path, split_name, class_name)
            _, _, images = list(os.walk(split_class_path))[0]
            print(os.path.join(split_name, class_name), len(images))

            class_path = os.path.join(path, class_name)
            if not os.path.exists(class_path): os.makedirs(class_path)
            for image in images:
                src = os.path.join(split_class_path, image)
                dest = os.path.join(class_path, image)
                shutil.move(src, dest)

        # Remove current split-dir
        split_path = os.path.join(path, split_name)
        shutil.rmtree(split_path)
        
    # Output results
    for class_name in class_names:
        class_path = os.path.join(path, class_name)
        _, _, class_images = list(os.walk(class_path))[0]
        print("->", class_name, len(class_images))



if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_const', const=True)
    parser.add_argument('--path', nargs='?', type=str, required=True)

    parser.add_argument('--src_path_isic', nargs='?', type=str)
    parser.add_argument('--src_path_praxis', nargs='?', type=str)
    parser.add_argument('--src_path_mednode', nargs='?', type=str)
    parser.add_argument('--src_path_ph2', nargs='?', type=str)

    parser.add_argument('--sort', action='store_const', const=True)
    parser.add_argument('--sort_prefer_masked', action='store_const', const=True)
    parser.add_argument('--sort_restrictive', action='store_const', const=True)

    parser.add_argument('--resize', action='store_const', const=True)
    parser.add_argument('--size', nargs='?', type=int, default=250)
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--augbalance', nargs='?', type=float)
    group.add_argument('--balance', action='store_const', const=True)
    group.add_argument('--revert_balance', action='store_const', const=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--split', action='store_const', const=True)
    group.add_argument('--revert_split', action='store_const', const=True)

    args = parser.parse_args()


    # Sanity Checks
    dataset_supplied = args.src_path_isic or args.src_path_praxis or args.src_path_mednode or args.src_path_ph2
    if args.sort and not dataset_supplied:
        raise ValueError(f"No dataset path was supplied.")

    if args.sort and args.src_path_isic and not os.path.isdir(args.src_path_isic):
        raise ValueError(f"Given path {args.src_path_isic} doesn't exist.")
    if args.sort and args.src_path_praxis and not os.path.isdir(args.src_path_praxis):
        raise ValueError(f"Given path {args.src_path_praxis} doesn't exist.")
    if args.sort and args.src_path_mednode and not os.path.isdir(args.src_path_mednode):
        raise ValueError(f"Given path {args.src_path_mednode} doesn't exist.")
    if args.sort and args.src_path_ph2 and not os.path.isdir(args.src_path_ph2):
        raise ValueError(f"Given path {args.src_path_ph2} doesn't exist.")
    
    if not (args.sort or args.resize or args.split or args.revert_split or args.balance or (args.augbalance is not None) or args.revert_balance):
        raise ValueError(f"No actions given to execute.")
    
    if args.balance and (args.balance <= .0 or args.balance > 1):
        raise ValueError(f"Balance-Ratio must be in (0, 1].")


    # Actual Actions
    if args.sort: sort_datasets(args)
    if args.resize: resize_images(args.path, args.size)
    if args.revert_split: revert_split(args.path)
    if args.split: split(args.path)
    # IMPORTANT: Test-set should never be balanced or augmented
    if args.revert_balance: revert_balance(args.path)
    if args.balance: 
        balance(os.path.join(args.path, 'training'))
        balance(os.path.join(args.path, 'validation'))
    if args.augbalance is not None: 
        augment_and_balance(os.path.join(args.path, 'training'), args.augbalance)
        balance(os.path.join(args.path, 'training'))  # This will only restore rundungsfehler (e.g. duplicating 20 images)
        balance(os.path.join(args.path, 'validation'))
    

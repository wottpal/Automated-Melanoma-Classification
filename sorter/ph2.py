# -*- coding: utf-8 -*-
"""
Sorts the PH2-Dataset into classes.
TODO IMPORTANT: Ensure normal XOR masked version is used only
"""

import cv2
import csv
import os
import math
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from .isic import mask_image


def dataset_info_from_csv(filepath):
    filenames = []
    diagnoses = []
    num_skipped = 0
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            try:
                # Determine diagnosis
                diagnosis = None
                if row[1].lower() == "x": 
                    diagnosis = 'benign'        # Common Nevus
                elif row[2].lower() == "x":
                    # diagnosis = 'malignant'   # Atypical Nevus (Removed because unsure if really histologiebasierend)
                    num_skipped += 1
                    continue
                elif row[3].lower() == "x": 
                    diagnosis = 'malignant'     # Melanoma 
                if not diagnosis: raise ValueError()
                
                # Append filename and according diagnosis
                filenames.append(row[0])
                diagnoses.append(diagnosis)
            except: continue

    print(f"Skipped {num_skipped} PH2-Images")

    return filenames, diagnoses


def apply(src_path, dest_path, restrictive_assignment, prefer_masked_version):
    # Read filenames and diagnoses from CSV
    csv_path = os.path.join(src_path, 'PH2_dataset_custom.csv')
    if not os.path.isfile(csv_path): raise ValueError(f"Can't find '{csv_path}'")
    filenames, diagnoses = dataset_info_from_csv(csv_path)

    for filename, diagnosis in tqdm(zip(filenames, diagnoses), total=len(filenames)):
        # Determine image- and mask-filepaths
        image_filepath = os.path.join(src_path, "PH2 Dataset images", filename, f"{filename}_Dermoscopic_Image", f"{filename}.bmp")
        mask_filepath = os.path.join(src_path, "PH2 Dataset images", filename, f"{filename}_lesion", f"{filename}_lesion.bmp")

        if not os.path.isfile(image_filepath) or not os.path.isfile(mask_filepath):
            print(f"Couldn't find PH2-image '{image_filepath}' or PH2-mask '{mask_filepath}'.")
            continue

        # Determine Destination
        dest_dir = os.path.join(dest_path, diagnosis)
        dest = os.path.join(dest_dir, f"{filename}.jpg")
        if not os.path.exists(dest_dir): os.makedirs(dest_dir)

        # Save masked version only if `prefer_masked_version` is True
        masked_image = None
        if prefer_masked_version:
            masked_image = mask_image(image_filepath, mask_filepath)

        if masked_image is not None:
            masked_dest = os.path.join(dest_dir, f"{filename}_MASKED.jpg")
            cv2.imwrite(masked_dest, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        else:
            image = cv2.imread(image_filepath)
            cv2.imwrite(dest, image)


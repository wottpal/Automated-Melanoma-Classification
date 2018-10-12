# -*- coding: utf-8 -*-
"""
Sorts the MED-NODE-dataset into classes. (Actually it's already sorted, just copying it)

NOTE: I've removed the `naevi` folder cause it's unsure what they've categorized to be a neavi and including e.g. DZN into `benign` would contradict my classification efforts.
"""

import os
import shutil
from tqdm import tqdm



def apply(src_path, dest_path, restrictive_assignment):
    filepaths = []
    for root, _, files in os.walk(src_path):
        filepaths += [os.path.join(root, x) for x in files]

    for filepath in tqdm(filepaths):
        _, filename = os.path.split(filepath)
        if not filename.lower().endswith(('.jpg', '.jpeg')): continue

        # Copy Image in diagnosis-directory
        dest_dir = os.path.join(dest_path, 'malignant')
        dest = os.path.join(dest_dir, filename)
        if not os.path.exists(dest_dir): os.makedirs(dest_dir)
        shutil.copyfile(filepath, dest)

# -*- coding: utf-8 -*-
"""Sorts the Hautpraxis-dataset collected by the `Praxis für Venen- und Hauterkrankungen Jena` into classes.
The following subfolders of the dataset will be sorted:
    1.Dysplastischer NZN 
    2.Benigne
    3.Maligne
    5.Basaliom
    6.Spinaliom
"""

import os
import shutil
from tqdm import tqdm
from PIL import Image
import uuid



def apply(src_path, dest_path, restrictive_assignment):
    images = []
    for root, subdirs, files in os.walk(src_path):
        if root == src_path: continue

        # Skip Directories with >1 or <1 image
        image_files = list(filter(lambda x: x.lower().endswith(('.jpg', '.jpeg', '.png')), files))
        # skipped = list(filter(lambda x: x not in set(image_files), set(files)))
        # print(f"SKIPPED = {skipped}")

        if len(image_files) > 1: print(image_files)
        if len(image_files) != 1: continue
        
        image_filename = image_files[0]
        image_path = os.path.join(root, image_filename)
        images.append(tuple((image_filename, image_path, root)))

    # Define Categorization
    if restrictive_assignment:
        BENIGN_DIAGNOSES = ("benigne")
        MALIGN_DIAGNOSES = ("maligne", "basaliom", "spinaliom")
    else:
        BENIGN_DIAGNOSES = ("benigne", "dysplastischer nzn")
        MALIGN_DIAGNOSES = ("maligne", "basaliom", "spinaliom")

    # Determine Actual Diganosis
    num_skipped = 0
    for image_filename, image_path, root in tqdm(images):
        diagnosis_dir = os.path.split(os.path.dirname(root))[1]
        if diagnosis_dir.lower().endswith(BENIGN_DIAGNOSES):
            diagnosis = "benign"
        elif diagnosis_dir.lower().endswith(MALIGN_DIAGNOSES):
            diagnosis = "malignant"
        else: 
            num_skipped += 1
            continue
        
        # Determine destination directory and new filename (append hash because of duplicate names)
        dest_dir = os.path.join(dest_path, diagnosis)
        if not os.path.exists(dest_dir): os.makedirs(dest_dir)
        short_hash = str(uuid.uuid4())[:4]
        dest = os.path.join(dest_dir, os.path.splitext(image_filename)[0] + f"-{short_hash}")

        # Copy JPG-Images
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            shutil.copyfile(image_path, dest + '.jpg')

        # Copy PNG-Images to JPGs
        elif image_path.lower().endswith(('.png')):
            print(f"Changing Format of {image_path}")
            im = Image.open(image_path)
            im.load()
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[3])
            background.save(dest + '.jpg', 'JPEG', quality=80)
            
        else: 
            num_skipped += 1
            continue
    
    print(f"Skipped {num_skipped} Hautpraxis-Images")


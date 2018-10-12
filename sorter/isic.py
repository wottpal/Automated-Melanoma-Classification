# -*- coding: utf-8 -*-
"""
Sorts the ISIC-dataset downloaded by the `ISIC-Archive-Downloader` (with "Images" and "Description" folders) into classes.
"""

import os
import shutil
import json
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np



def mask_image(image_path, segmentation_path):
    # Load Image and Segmentation-Image
    mask = cv2.imread(segmentation_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make 2px black border around image to prevent artifacts
    top, bottom, left, right = [2] * 4
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Masked Image
    try: masked_image = cv2.bitwise_and(image, image, mask=mask)
    except: return None

    # Find Contours & Bounding Box with OpenCV
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Crop & Rotate Bounding Box evenly
    W = rect[1][0]
    H = rect[1][1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    if angle < -45:
        angle+=90

    center = (int((x1+x2)/2), int((y1+y2)/2))
    mult = 1.15
    size = (int(mult*(x2-x1)),int(mult*(y2-y1)))

    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    cropped = cv2.getRectSubPix(masked_image, size, center)    
    cropped = cv2.warpAffine(cropped, M, size)

    # Fit into square
    square_size = max(size[0], size[1])
    x_start, y_start = int((square_size - size[0]) / 2), int((square_size - size[1]) / 2)
    x_end, y_end = x_start + size[0], y_start + size[1]
    cropped_square = np.zeros((square_size, square_size, 3)).astype(np.uint8)
    if size[0] < size[1]: cropped = np.rot90(cropped)
    cropped_square[x_start:x_end][y_start:y_end][:] += cropped

    # fig = plt.figure()
    # fig.add_subplot(2, 2, 1)
    # plt.imshow(image)
    # fig.add_subplot(2, 2, 2)
    # plt.imshow(mask)
    # fig.add_subplot(2, 2, 3)
    # plt.imshow(masked_image)
    # fig.add_subplot(2, 2, 4)
    # plt.imshow(cropped_square)
    # plt.show()

    return cropped_square



def apply(src_path, dest_path, restrictive_assignment, prefer_masked_version):
    isic_src_images_path = os.path.join(src_path, "Images")
    isic_src_json_path = os.path.join(src_path, "Descriptions")
    isic_src_segmentation_path = os.path.join(src_path, "Segmentation")

    filepaths = []
    for root, _, files in os.walk(isic_src_images_path):
        filepaths += [os.path.join(root, x) for x in files]

    segmentation_filepaths = []
    for root, _, files in os.walk(isic_src_segmentation_path):
        segmentation_filepaths += [os.path.join(root, x) for x in files]

    num_skipped, num_masked = 0, 0
    for filepath in tqdm(filepaths):
        _, filename = os.path.split(filepath)
        if not filename.lower().endswith(('.jpg', '.jpeg')): continue

        json_path = os.path.join(isic_src_json_path, os.path.splitext(filename)[0])
        try:
            json_data = json.load(open(json_path))

            # Look for `benign_malignant`-field or classify `diagnosis` field
            if not ("meta" in json_data and "clinical" in json_data["meta"]): continue
            clinical_data = json_data["meta"]["clinical"]

            if "benign_malignant" in clinical_data:
                diagnosis = clinical_data["benign_malignant"]

            elif "diagnosis" in clinical_data:
                diagnosis = clinical_data["diagnosis"]

                # Define Categorization (if malign/benign not given, of course)
                if restrictive_assignment:
                    BENIGN_DIAGNOSES = ("pigmented benign keratosis", "dermatofibroma", "vascular lesion")
                    MALIGN_DIAGNOSES = ("basal cell carcinoma", "squamous cell carcinoma")
                else:
                    BENIGN_DIAGNOSES = ("pigmented benign keratosis", "dermatofibroma", "vascular lesion")
                    MALIGN_DIAGNOSES = ("basal cell carcinoma", "squamous cell carcinoma", "actinic keratosis")

                # Determine Actual Diganosis
                if diagnosis.lower().endswith(BENIGN_DIAGNOSES):
                    diagnosis = "benign"
                elif diagnosis.lower().endswith(MALIGN_DIAGNOSES):
                    diagnosis = "malignant"
                else: 
                    num_skipped += 1
                    continue
            else: 
                num_skipped += 1
                continue
        except: 
            num_skipped += 1
            continue


        if not diagnosis or not diagnosis.lower() in ["benign", "malignant"]: 
            num_skipped += 1
            continue

        # Create Masked Version of Image if Segmentation Available
        segmentation, masked_image = None, None
        if prefer_masked_version:
            for segmentation_filepath in segmentation_filepaths:
                _, segmentation_filename = os.path.split(segmentation_filepath)
                if not segmentation_filename.lower().endswith(('.png')): continue
                if not segmentation_filename.startswith((os.path.splitext(filename)[0])): continue
                segmentation = segmentation_filepath
            if segmentation is not None:
                masked_image = mask_image(filepath, segmentation)

        # Filter out SONIC images without Segmentation
        if "dataset" in json_data and "name" in json_data["dataset"]:
            dataset_name = json_data["dataset"]["name"]
            if dataset_name.lower() == "sonic" and masked_image is None: 
                num_skipped +=1
                continue
            elif dataset_name.lower() != "sonic": # TODO IMPORTANT
                num_skipped +=1
                continue


        # Save Either Masked or Normal version
        dest_dir = os.path.join(dest_path, diagnosis)
        dest = os.path.join(dest_dir, filename)
        if not os.path.exists(dest_dir): os.makedirs(dest_dir)

        if masked_image is not None:
            num_masked += 1
            masked_path = os.path.join(dest_dir, os.path.splitext(filename)[0] + '_MASKED.jpg')
            cv2.imwrite(masked_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        else:
            shutil.copyfile(filepath, dest)


    print(f"Skipped {num_skipped} ISIC-Images (FYI: SONIC is around 9250)")
    print(f"Masked {num_masked} ISIC-Images")



    

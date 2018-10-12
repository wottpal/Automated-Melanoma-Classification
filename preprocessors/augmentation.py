# -*- coding: utf-8 -*-
"""Creates different versions with different lightning."""

import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmenters import Augmenter, meta
import six.moves as sm
import random
import os 
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import argparse



# Main Routine
def apply(img):
    seq = iaa.Sequential([
        iaa.Sometimes(0.75, iaa.Add((-25, 25))),
        iaa.Sometimes(0.75, iaa.AddToHueAndSaturation((-3, 2))),
        iaa.Sometimes(0.75, iaa.ContrastNormalization((0.8, 1.3))),
        iaa.Sometimes(0.75, iaa.Fliplr(1)),
        iaa.Sometimes(0.75, iaa.Flipud(1)),
        iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0.025, 0.1))),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=0.005*255)),
        iaa.Sometimes(0.5, iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-45, 45),
            shear=(-5, 5),
        ))
    ], random_order=True)

    img = seq.augment_image(img)

    return img



# Plotting Helpers (Can't embedd other submodule, shame on you Python!)
def plt_show():
    """Workaround for nasty macOS-bug: https://github.com/matplotlib/matplotlib/issues/9637#issuecomment-363682221"""
    try:
        plt.show()
    except UnicodeDecodeError:
        plt_show()

def plot_images(images):
    n = len(images)
    rows = int(math.sqrt(n))
    cols = n // rows
    fig = plt.figure(figsize=(2.5 * cols, 2.5 * rows))
    for i in range(1, n+1):
        image = images[i-1]
        fig.add_subplot(rows, cols, i)
        plt.xticks([]),plt.yticks([])
        plt.imshow(image)
    fig.tight_layout()
    plt_show()

def plot_bgr_images(images):
    conversionMode = lambda img: cv2.COLOR_GRAY2RGB if len(img.shape) == 2 else cv2.COLOR_BGR2RGB
    images = [cv2.cvtColor(img, conversionMode(img)) for img in images]
    plot_images(images)



if __name__ == "__main__":   
    """
    Execute this script as main for debugging purposes. It does not modify/write anything.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='?', type=str, default="./data/training")
    args = parser.parse_args()
    
    filepaths = []
    for root, _, files in os.walk(args.path):
        filepaths += [os.path.join(root, x) for x in files]

    random.shuffle(filepaths)
    
    for filepath in tqdm(filepaths):
        if not filepath.lower().endswith(('.jpg', '.jpeg')): continue
        image = cv2.imread(filepath)

        # Create Augmented Versions
        augmented = []
        for _ in range(5):
            augmented.append(apply(image))

        plot_bgr_images([image] + augmented)

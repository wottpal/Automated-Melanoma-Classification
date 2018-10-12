# -*- coding: utf-8 -*-
"""Factory to generate different data generators. 
NOTE: This script can't be executed directly due to parent modul import limitations of python."""

from helpers import io, plotter
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from keras.preprocessing.image import *
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt



# def x_train_samples(train_path, image_shape, max_samples):
#     """This returns a small but statistically representative sample of the training data.
#     It's used to fit the ImageDataGenerator which is mandatory is using for example `zca_whitening`.
#     Reference: https://stackoverflow.com/a/46709583/1381666"""
#     classes, images = io.subdirs_and_files(train_path)

#     len_images = [len(x) for x in images]
#     samples_per_class = [int(max_samples * x / sum(len_images)) for x in len_images]
#     samples_per_class = [min(x, len_images[idx]) for idx, x in enumerate(samples_per_class)]

#     samples = [np.random.choice(images[idx], x, replace=False) for idx, x in enumerate(samples_per_class)]
#     samples = [os.path.join(classes[idx], item) for idx, sublist in enumerate(samples) for item in sublist]

#     sample_images = []
#     for path in samples:
#         try:
#             img = cv2.imread(os.path.join(train_path, path))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = img.reshape(image_shape).astype("float32")
#             sample_images.append(img)
#         except: continue
#     sample_images = np.asarray(sample_images)

#     return sample_images



def v1WithoutAugmentation(data_path, target_size, batch_size, preprocessor):
    train_path = os.path.join(data_path, 'training')
    validation_path = os.path.join(data_path, 'validation')

    train_idg = ImageDataGenerator(preprocessing_function=preprocessor)
    validation_idg = ImageDataGenerator(preprocessing_function=preprocessor)

    train_gen = train_idg.flow_from_directory(
        train_path, 
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=42
    )
    validation_gen = validation_idg.flow_from_directory(
        validation_path, 
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=42
    )

    return (train_gen, validation_gen)


def v2AugmentedMinimal(data_path, target_size, batch_size, preprocessor):
    train_path = os.path.join(data_path, 'training')
    validation_path = os.path.join(data_path, 'validation')

    train_idg = ImageDataGenerator(
        preprocessing_function=preprocessor,
        featurewise_center=False,
        samplewise_center=False,
        rotation_range=180,
        width_shift_range=.1,
        height_shift_range=.1,
        shear_range=.1,
        zoom_range=.1,
        zca_whitening=False,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode = "constant",
        cval=0.0
    )
    validation_idg = ImageDataGenerator(preprocessing_function=preprocessor)

    train_gen = train_idg.flow_from_directory(
        train_path, 
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=42
    )
    validation_gen = validation_idg.flow_from_directory(
        validation_path, 
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=42
    )

    return (train_gen, validation_gen)


def v3AugmentedMore(data_path, target_size, batch_size, preprocessor):
    train_path = os.path.join(data_path, 'training')
    validation_path = os.path.join(data_path, 'validation')

    train_idg = ImageDataGenerator(
        preprocessing_function=preprocessor,
        featurewise_center=True,
        samplewise_center=True,
        rotation_range=360,
        width_shift_range=.15,
        height_shift_range=.15,
        shear_range=.15,
        zoom_range=.15,
        zca_whitening=False,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode = "constant",
        cval=0.0
    )
    validation_idg = ImageDataGenerator(preprocessing_function=preprocessor)

    train_gen = train_idg.flow_from_directory(
        train_path, 
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=42
    )
    validation_gen = validation_idg.flow_from_directory(
        validation_path, 
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=42
    )

    return (train_gen, validation_gen)



# All Implemented Generators
GENERATORS = [
    v1WithoutAugmentation,
    v2AugmentedMinimal,
    v3AugmentedMore
]

def count():
    return len(GENERATORS)

def get(i):
    return GENERATORS[i]

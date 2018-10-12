# -*- coding: utf-8 -*-
"""Ability to predict a single or multiple images with a trained model."""

from helpers import model_loader
from factories import model_factory
import cam 
import keras
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys
import sty
import argparse
from tqdm import tqdm



def predict_image(model_path, model_type, image, image_size = 250, activation_maps = False):
    """
    Returns the prediction for the given image-data and model. 
    """
    model = model_loader.get_model(model_path)
    _, preprocessor = model_factory.get(model_type)

    # Ensure image has correct size and rescale it to floats
    image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image = preprocessor(image)
    image = image.reshape((1, image_size, image_size, 3))

    print("\nPredict Image..")
    predictions = model.predict(image, verbose=1)[0]
    predicted_class = np.argmax(predictions)
    single_value_prediction = predictions[predicted_class] if predictions[1] > predictions[0] else 1 - predictions[predicted_class]

    heatmap_evolutions = None
    if activation_maps == 'all':
        print("\nBuild All Class Activation Heatmaps..")
        heatmap_evolutions = cam.class_activation_map_evolutions(model, len(predictions), image, 250)
    if activation_maps == 'last':
        print("\nBuild Last-Layer Class Activation Heatmaps..")
        heatmap_evolutions = cam.class_activation_maps(model, len(predictions), image, 250)
    
    print(sty.ef.dim + f"[predictions={predictions}, single_value_prediction={single_value_prediction}]\n" + sty.rs.all)
    return predictions.tolist(), single_value_prediction, predicted_class, heatmap_evolutions



def predict_image_path(model_path, model_type, image_path, image_size = 250, activation_maps = False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return predict_image(model_path, model_type, image, image_size, activation_maps)



def predict_images_path(model_path, model_type, images_path, image_size = 250, limit = False):
    _, preprocessor = model_factory.get(model_type)

    print("\nLoad Image-Directory..")
    filepaths = []
    for root, _, files in os.walk(images_path):
        filepaths += [os.path.join(root, x) for x in files]

    # Load images into memory and preprocess
    images = []
    paths = []
    for filepath in tqdm(filepaths):
        _, filename = os.path.split(filepath)
        if not filename.lower().endswith(('.jpg', '.jpeg')): continue            
        if limit and len(images) >= limit: break

        try:
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
            image = preprocessor(image)
            image = image.reshape((image_size, image_size, 3))
            images.append(image)
            paths.append(filepath)
        except: continue
        
    images = np.asarray(images).reshape((len(images), image_size, image_size, 3))

    print("\nPredict Image-Directory..")
    model = model_loader.get_model(model_path)
    predictions = model.predict(images, verbose=1)

    single_value_predictions = []
    class_predictions = []
    for prediction in predictions:
        predicted_class = np.argmax(prediction)
        single_value_prediction = prediction[predicted_class] if prediction[1] > prediction[0] else 1 - prediction[predicted_class]
        class_predictions.append(predicted_class)
        single_value_predictions.append(single_value_prediction)
    
    return paths, predictions, single_value_predictions, class_predictions
    


if __name__ == "__main__":
    # TODO add "activations" option which plots heatmaps
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, nargs='?', required=True)
    parser.add_argument('--model_type', type=int, required=True, choices=range(0, model_factory.count()))
    parser.add_argument('--image_size', type=int, nargs='?', default=250)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str)
    group.add_argument('--images_path', type=str)
    args = parser.parse_args()

    if args.image_path:
        prediction = predict_image_path(args.model_path, args.model_type, args.image_path, args.image_size)
    else:
        prediction = predict_images_path(args.model_path, args.model_type, args.images_path, args.image_size)

    print(prediction)

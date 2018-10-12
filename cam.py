# -*- coding: utf-8 -*-
"""
Generates Class Activation Maps
Reference: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
"""

from helpers import model_loader, plotter, io, colors
from factories import model_factory
import predict
import keras
from keras import backend as K
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys
import argparse
import imageio



def get_conv2d_layers(model):
    conv_layers = []
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == "Conv2D":
            conv_layers.append(layer)
    
    return conv_layers


def normalize_and_resize_heatmaps(heatmaps, image_size = 250):
    # Determine max of all given heatmaps
    global_max = 0
    for heatmap in heatmaps:
        global_max = max(np.max(heatmap), global_max)
    print(f"Global-Max = {global_max}")

    # Normalize all heatmaps by that max
    normalized_heatmaps = []
    for heatmap in heatmaps:
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / global_max if global_max != .0 else heatmap * 0
        heatmap = cv2.resize(heatmap, (image_size, image_size))
        normalized_heatmaps.append(heatmap)

    return normalized_heatmaps


def class_activation_map(model, layer, num_class, image, image_size = 250):
    class_output = model.output[:, num_class]

    grads = K.gradients(class_output, layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([image])  

    layer_output_size = layer.output.shape[-1]
    for i in range(layer_output_size):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    return heatmap


def class_activation_maps(model, num_classes, image, image_size = 250):
    # Find last conv2d layer
    conv_layers = get_conv2d_layers(model)
    if not conv_layers: return None
    last_conv_layer = conv_layers[-1]

    # Generate activation heatmaps for each class
    heatmaps = []
    for i in range(num_classes):
        heatmap = class_activation_map(model, last_conv_layer, i, image, image_size)
        heatmaps.append(heatmap)
    
    # Normalize Heatmaps by global max and resize
    heatmaps = normalize_and_resize_heatmaps(heatmaps, image_size)
    
    return heatmaps


def class_activation_map_evolutions(model, num_classes, image, image_size = 250):
    # Find all conv2d layers
    conv_layers = get_conv2d_layers(model)
    print(f"Found {len(conv_layers)} Conv2D-Layers")
    if not conv_layers: return None

    # Generate activation heatmaps for each class & Normalize
    heatmap_evolutions = []
    for _ in range(num_classes):
        heatmap_evolutions.append([None] * len(conv_layers))

    for l in range(len(conv_layers)):
        heatmaps_per_layer = []
        for c in range(num_classes):
            heatmap = class_activation_map(model, conv_layers[l], c, image, image_size)
            heatmaps_per_layer.append(heatmap)
        normalized_heatmaps_per_layer = normalize_and_resize_heatmaps(heatmaps_per_layer, image_size)
        for c, heatmap in enumerate(normalized_heatmaps_per_layer):
            heatmap_evolutions[c][l] = heatmap

    return heatmap_evolutions


def superimpose_heatmap_on_image(image, heatmap):
    heatmap = np.uint8(255 * (1 - heatmap))
    heatmap = np.maximum(heatmap, 0)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(image, 1.0, heatmap, 0.6, 0)

    return superimposed


def superimpose_weighted_heatmap_on_image(image, heatmaps, predictions):
    return image
    # heatmap = heatmaps[0] * predictions[0] + heatmaps[1] * predictions[1]
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = colors.applyTrafficLightColorMap(heatmap)
    # superimposed = cv2.addWeighted(image, 1.0, heatmap, 0.7, 0)

    # return superimposed


def label_image(image, text):
    labeled_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (5, image.shape[0] - 10)
    cv2.putText(labeled_image, text, pos, font, .6, (0,0,0), 3)
    cv2.putText(labeled_image, text, pos, font, .6, (255,255,255), 1)
    return labeled_image


def save_class_activation_map_evolutions(model_path, model_type, num_classes, image_path, image, image_size=250, output_dir="./results/cam/"):
    # Generate Prediction & CAM-Evolutions
    # NOTE: Ironically I use the function of another module which then calls `class_activation_map_evolutions`.
    #       But that's how it is... :D
    predictions, prediction, predicted_class, all_map_evolutions = predict.predict_image(model_path, model_type, image, image_size, 'all')

    # Superimpose weighted & labeled heatmaps on image
    all_map_evolutions_superimposed = []
    for num_class in range(0, num_classes):
        class_prediction = predictions[num_class]

        map_evolutions_superimposed = [ label_image(image, "0 - Input") ]
        weighted_map_evolutions_superimposed = [ label_image(image, "0 - Input") ]

        for idx, heatmap in enumerate(all_map_evolutions[num_class]):
            superimposed_heatmap = superimpose_heatmap_on_image(image, heatmap)
            superimposed_heatmap = label_image(superimposed_heatmap, f"{idx+1} - Conv2D")
            map_evolutions_superimposed.append(superimposed_heatmap)

            superimposed_weighted_heatmap = superimpose_heatmap_on_image(image, heatmap * class_prediction)
            superimposed_weighted_heatmap = label_image(superimposed_weighted_heatmap, f"{idx+1} - Conv2D (Weighted)")
            weighted_map_evolutions_superimposed.append(superimposed_weighted_heatmap)

        all_map_evolutions_superimposed.append([map_evolutions_superimposed, weighted_map_evolutions_superimposed])


    # Create Unique Output Directory
    output_dir = io.next_path(output_dir, create_dirs=True)

    # Generate & Save Weighted Heatmap
    # last_heatmaps = [all_map_evolutions[0][-1], all_map_evolutions[1][-1]]
    # superimposed_weighted_heatmap = superimpose_weighted_heatmap_on_image(image, last_heatmaps, predictions)
    # output_path = os.path.join(output_dir, f'CAM_weighted.png')
    # cv2.imwrite(output_path, cv2.cvtColor(superimposed_weighted_heatmap, cv2.COLOR_RGB2BGR))

    print(f"Save Animated GIF & Seperate PNGs..")
    for num_class in range(0, num_classes):
        output_dir_frames = os.path.join(output_dir, f"CAM_class_{num_class}_frames/")
        if not os.path.exists(output_dir_frames): os.makedirs(output_dir_frames)

        heatmaps_seq = all_map_evolutions_superimposed[num_class][0]
        weighted_heatmaps_seq = all_map_evolutions_superimposed[num_class][1]

        # Seperate PNGs
        for idx in range(len(heatmaps_seq)):
            heatmap = heatmaps_seq[idx]
            output_path = os.path.join(output_dir_frames, f'CAM_class_{num_class}_frame_{idx}.png')
            # cv2.imwrite(output_path, heatmap)
            cv2.imwrite(output_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

            weighted_heatmap = weighted_heatmaps_seq[idx]
            weighted_output_path = os.path.join(output_dir_frames, f'CAM_class_{num_class}_weighted_frame_{idx}.png')
            # cv2.imwrite(weighted_output_path, weighted_heatmap)
            cv2.imwrite(weighted_output_path, cv2.cvtColor(weighted_heatmap, cv2.COLOR_RGB2BGR))

        # Animated GIF
        output_path = os.path.join(output_dir, f'CAM_class_{num_class}_animated.gif')
        imageio.mimsave(output_path, heatmaps_seq, duration=0.5)

        output_path = os.path.join(output_dir, f'CAM_class_{num_class}_weighted_animated.gif')
        imageio.mimsave(output_path, weighted_heatmaps_seq, duration=0.5)

    
    # Write Info-File
    info_path = os.path.join(output_dir, "_INFO.txt")
    predicted_class_name = "Benign" if predicted_class == 0 else "Malignant"
    with open(info_path, "w") as text_file:
        print(f"Image-Path: {image_path}", file=text_file)
        print(f"Model-Path: {model_path}", file=text_file)
        print(f"Model-Type: {model_type}", file=text_file)
        print(f"Predicted Class: {predicted_class} ({predicted_class_name})", file=text_file)
        print(f"Exact Predictions (Weight Factors): {predictions}", file=text_file)
        print(f"Single Value Prediction: {prediction}", file=text_file)
        print(f"\nCommand:", file=text_file)
        print(f'python cam.py --model_path "{model_path}" --model_type {model_type} --image_path "{image_path}"', file=text_file)

    return output_dir



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str, nargs='?')
    parser.add_argument('--model_type', required=True, type=int, choices=range(0, model_factory.count()))
    parser.add_argument('--image_path', required=True, type=str)
    parser.add_argument('--output_dir', type=str, nargs='?', default="./results/cam/")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--image_size', type=int, nargs='?', default=250)
    args = parser.parse_args()

    # Load Image
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate & Save CAM-Evolutions
    save_class_activation_map_evolutions(args.model_path, args.model_type, args.num_classes, args.image_path, image, image_size=args.image_size, output_dir=args.output_dir)
    


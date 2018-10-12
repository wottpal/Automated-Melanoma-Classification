# -*- coding: utf-8 -*-

from helpers import plotter
from factories import generator_factory, model_factory
import os
import argparse
import numpy as np



# import numpy as np
# from keras.preprocessing.image import ImageDataGenerator

# def cause_leak():
#     idg = ImageDataGenerator(zca_whitening = True)
#     random_sample = np.random.random((1, 250, 250, 3))
#     idg.fit(random_sample)
# cause_leak()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, required=True, choices=range(0, model_factory.count()))
    parser.add_argument('--generator_type', type=int, required=True, choices=range(0, generator_factory.count()))
    parser.add_argument('--data_path', type=str, nargs='?', default='./data')
    parser.add_argument('--image_size', type=int, nargs='?', default=250)
    parser.add_argument('--batch_size', type=int, nargs='?', default=6)
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        raise ValueError(f"Given directory doesn't exist.")

    _, preprocessor = model_factory.get(args.model_type)
    generators = generator_factory.get(args.generator_type)
    train_gen, val_gen = generators(args.data_path, (args.image_size, args.image_size), args.batch_size, preprocessor)
    
    for data_batch, data_batch_labels in train_gen:
        print(data_batch[0])
        print(data_batch_labels)
        print(f"Shape: {data_batch.shape}")        

        # Revert Imagenet Preprocessing Function
        # data_batch = np.asarray(data_batch).astype("float32")
        data_batch += 127.5
        data_batch /= 255.
    
        plotter.plot_bgr_images(data_batch)
# -*- coding: utf-8 -*-
"""Builds and (re)trains the CNN."""

from helpers import io, prowl, metrics, model_loader
from helpers.weighted_loss import weights_are_equal, weights_to_dict, WeightedCategoricalCrossEntropy
from factories import generator_factory, model_factory, optimizer_factory
import tensorflow as tf
import keras
from keras import layers, models, optimizers
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np
import argparse
import sty

K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf')



def build_callbacks(args):
    callbacks = []
    if not args.log and not args.save and not args.notify: return callbacks

    print("\nBuild Callbacks..")

    if args.log:
        print("TensorBoard")
        log_path = io.next_path('./results/logs/', create_dirs=True)
        callbacks.append(TensorBoard(
            log_dir=log_path, 
            write_graph=True, 
            write_images=True
        ))

    if args.info:
        print("Info-File")
        info_path = io.next_path('./results/logs/', '_INFO.txt', create_dirs=True, append=args.log)
        with open(info_path, "w") as txt:
            if args.retrain_latest_model_from_epoch is not None:
                print(f"Retrain from Epoch: {args.retrain_latest_model_from_epoch}\n", file=txt)
            print(f"Epochs: {args.epochs}", file=txt)
            print(f"Batch-Size: {args.batch_size}", file=txt)
            print(f"Model: {args.model_type}", file=txt)
            print(f"Optimizer: {args.optimizer_type}", file=txt)
            print(f"Generator: {args.generator_type}", file=txt)
            print(f"Loss-Weights: {args.loss_weights}", file=txt)
            print(f"Clas-Weights: {args.class_weights}", file=txt)
            if args.earlystop:
                print(f"\nEarlystop-Monitor: {args.earlystop_monitor}", file=txt)
                print(f"Earlystop-Patience: {args.earlystop_patience}", file=txt)

    if args.save:
        print("Save Model-Checkpoints")
        models_dir = './results/models/'
        # NOTE: Even though it would be nice to give the saved models nicer names 
        #       (e.g. including number of epoch), this is totally not recommended
        #       as it saves tons of new models what quickly fills any disk.
        callbacks.append(ModelCheckpoint(
            filepath=io.next_path(models_dir, 'model-best-val_acc.h5', create_dirs=True), 
            save_best_only=True,
            monitor='val_acc', 
            mode='auto',
            verbose=1, 
        ))
        callbacks.append(ModelCheckpoint(
            filepath=io.next_path(models_dir, 'model-best-val_loss.h5', append=True), 
            save_best_only=True,
            monitor='val_loss', 
            mode='auto',
            verbose=1, 
        ))

    if args.earlystop:
        print("Enable Early Stopping")
        callbacks.append(EarlyStopping(
            monitor=args.earlystop_monitor, 
            patience=args.earlystop_patience, 
            mode='auto',
            verbose=1,
        ))
        
    if args.notify:
        print("Send Prowl-Notifications")
        callbacks.append(prowl.NotificationCallback())

    return callbacks


def build_model(args):
    loss_weights = None if weights_are_equal(args.loss_weights) else weights_to_dict(args.loss_weights)
    loss = 'categorical_crossentropy' if loss_weights is None else WeightedCategoricalCrossEntropy(weights_to_dict(args.loss_weights))
    optimizer = optimizer_factory.get(args.optimizer_type)
    model, preprocessor = model_factory.get(
        args.model_type,
        input_shape=(args.image_size, args.image_size, 3),
        optimizer=optimizer,
        metrics=['acc'],#, metrics.precision, metrics.recall],
        loss=loss
    )
    print(sty.ef.dim + f"[model={args.model_type}, optimizer={optimizer.__class__.__name__} ({args.optimizer_type}), loss_weights={loss_weights}]\n" + sty.rs.all)

    return model, preprocessor


def build_generators(args, preprocessor=0):
    print("\n\nScan Data-Directory..")
    print(sty.ef.dim + f"[data_path={args.data_path}]\n" + sty.rs.all)
    io.scan_directory(args.data_path)    

    print("\n\nBuild ImageDataGenerators..")
    generator = generator_factory.get(args.generator_type)
    print(sty.ef.dim + f"[generator={generator.__name__} ({args.generator_type}), image_size={args.image_size}]\n" + sty.rs.all)
    
    return generator(args.data_path, (args.image_size, args.image_size), args.batch_size, preprocessor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, nargs='?', default='./data')
    parser.add_argument('--image_size', type=int, nargs='?', default=250)
    parser.add_argument('--batch_size', type=int, nargs='?', default=32)
    parser.add_argument('--epochs', type=int, nargs='?', default=1)
    parser.add_argument('--model_type', type=int, required=True, choices=range(0, model_factory.count()))
    parser.add_argument('--generator_type', type=int, required=True, choices=range(0, generator_factory.count()))
    parser.add_argument('--optimizer_type', type=int, required=True, choices=range(0, optimizer_factory.count()))
    parser.add_argument("--class_weights", nargs=2, type=float, default=[1.0, 1.0])
    parser.add_argument("--loss_weights", nargs=2, type=float, default=[1.0, 1.0])
    parser.add_argument('--earlystop_monitor', type=str, nargs='?', default='val_acc')
    parser.add_argument('--earlystop_patience', type=int, nargs='?', default=5)
    parser.add_argument('--retrain_latest_model_from_epoch', type=int, nargs='?')
    parser.add_argument('-l', '--log', action='store_const', const=True)
    parser.add_argument('-i', '--info', action='store_const', const=True)
    parser.add_argument('-s', '--save', action='store_const', const=True)
    parser.add_argument('-e', '--earlystop', action='store_const', const=True)
    parser.add_argument('-n', '--notify', action='store_const', const=True)
    args = parser.parse_args()

    # Create Generators and Compute Class Weights
    print("\nBuild Generators & Determine Weights..")
    train_gen, val_gen = build_generators(args)
    train_len, val_len = len(train_gen.filenames), len(val_gen.filenames)
    class_weights = None if weights_are_equal(args.class_weights) else weights_to_dict(args.class_weights)
    print(sty.ef.dim + f"[validation_ratio={val_len/train_len:.2f}, class_weights={class_weights}]" + sty.rs.all)

    print("\nBuild Model..")
    model, preprocessor = build_model(args)
    callbacks = build_callbacks(args)
    model.summary()

    # Resume Training / Fine-Tune
    initial_epoch = 0
    if args.retrain_latest_model_from_epoch:
        lates_model_path = io.latest_files('./results/models/', ('.h5', '.h5df'), 1)
        if not lates_model_path: raise ValueError("Couldn't find existing model under './results/models/'. So can't continue training.")
        model = model_loader.get_model(lates_model_path[0])
        initial_epoch = args.retrain_latest_model_from_epoch

    print("\nFit Model..")
    print(sty.ef.dim + f"[epochs={args.epochs}, batch_size={args.batch_size}, steps_per_epoch={len(train_gen)}, validation_steps={len(val_gen)}]\n" + sty.rs.all)
    result = model.fit_generator(
        train_gen,
        initial_epoch = initial_epoch, 
        epochs = args.epochs + initial_epoch,
        validation_data = val_gen,
        class_weight = class_weights,
        callbacks = callbacks,
        verbose = 1
    )

    # print("\nSave Final Model..")
    # model_path = io.next_path('./results/models/', f'final-model_{args.epochs:03d}-epochs.h5', create_dirs=True, append=args.save)
    # model.save(model_path)
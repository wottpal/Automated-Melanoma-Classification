#!/bin/bash

# Insert specific training configuration here

BATCH_SIZE=16
EPOCHS=25
MODEL=10
GENERATOR=1
OPTIMIZER=13
CLASS_WEIGHTS=' 1.0 1.0 '
LOSS_WEIGHTS=' 1.0 1.0 '


# Load Model (for fine-tuning only)
# MODEL_URL= ...
# python acquire_model.py --model_url $MODEL_URL


# Load preprocessed Dataset
# DATASET_URL= ...
# python acquire_dataset.py --dataset_url $DATASET_URL


# Train
# python train.py -lsni --epochs $EPOCHS \
#                       --batch_size $BATCH_SIZE \
#                       --model_type $MODEL \
#                       --optimizer_type $OPTIMIZER \
#                       --generator_type $GENERATOR \
#                       --loss_weights $LOSS_WEIGHTS \
#                       --class_weights $CLASS_WEIGHTS


# Evaluate
# python evaluate.py -n --model_type $MODEL --latest_models 2 --keep_best_only


# Pull Results
# python backup_results.py


# IMPORTANT: Shutdown Instance
# sudo shutdown -h now

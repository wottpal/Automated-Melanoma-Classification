# -*- coding: utf-8 -*-
"""
Factory to generate different keras models.

TODO Add Models:
    - Xception
    - VGG19
    - Inception v4 (InceptionResNetV2)
    - DenseNet201
"""

import keras
from keras import layers, models, optimizers, regularizers
from keras.layers import *
from keras.models import *
from keras.applications.vgg16 import VGG16
import argparse



def V1_1(input_shape):
    model = models.Sequential()
    model.add(Convolution2D(32, (3,3), activation='elu', padding='same', input_shape=input_shape))        
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64, (3,3), activation='elu', padding='same'))
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(128, (3,3), activation='elu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), activation='elu', padding='same'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.7))
    model.add(Dense(2, activation='softmax'))

    return model


def V1_2(input_shape):
    model = models.Sequential()
    model.add(Convolution2D(32, (3,3), activation='elu', padding='same', use_bias=False, input_shape=input_shape))        
    model.add(BatchNormalization())        
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64, (3,3), activation='elu', padding='same', use_bias=False))
    model.add(BatchNormalization())        
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(128, (3,3), activation='elu', padding='same', use_bias=False))
    model.add(BatchNormalization())        
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), activation='elu', padding='same', use_bias=False))
    model.add(BatchNormalization())        
    model.add(MaxPooling2D((2,2)))

    l2 = regularizers.l2(l=0.01)

    model.add(Flatten())
    model.add(Dense(256, activation='elu', kernel_regularizer=l2))
    model.add(Dropout(0.7))
    model.add(Dense(2, activation='softmax'))

    return model


def V1_3(input_shape):
    model = models.Sequential()
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=input_shape))        
    model.add(BatchNormalization())      
    model.add(Activation('elu'))  
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization()) 
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())  
    model.add(Activation('elu'))        
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())  
    model.add(Activation('elu'))        
    model.add(MaxPooling2D((2,2)))

    l2 = regularizers.l2(l=0.01)

    model.add(Flatten())
    model.add(Dense(256, use_bias=False, kernel_regularizer=l2))
    model.add(BatchNormalization())  
    model.add(Activation('elu'))  
    model.add(Dropout(0.7))
    model.add(Dense(2, activation='softmax'))

    return model



def V1_4(input_shape):
    model = models.Sequential()
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=input_shape))        
    model.add(BatchNormalization())      
    model.add(Activation('relu'))  
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))         
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())  
    model.add(Activation('relu'))        
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())  
    model.add(Activation('relu'))        
    model.add(MaxPooling2D((2,2)))

    l2 = regularizers.l2(l=0.01)

    model.add(Flatten())
    model.add(Dense(256, use_bias=False, kernel_regularizer=l2))
    model.add(Activation('relu'))  
    model.add(Dropout(0.7))
    model.add(Dense(2, activation='softmax'))

    return model



def V2(input_shape):
    model = models.Sequential()
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=input_shape))
    model.add(BatchNormalization())   
    model.add(Activation('elu'))            
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))

    l2 = regularizers.l2(l=0.01)

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(Dropout(0.7))
    model.add(Dense(2, activation='softmax'))

    return model



def V3(input_shape):
    model = models.Sequential()
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=input_shape))
    model.add(BatchNormalization())   
    model.add(Activation('elu'))            
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))

    l2 = regularizers.l2(l=0.01)

    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=l2, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(Dropout(0.8))
    model.add(Dense(1024, kernel_regularizer=l2, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(Dropout(0.8))
    model.add(Dense(2, activation='softmax'))

    return model



def V4(input_shape):
    model = models.Sequential()
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=input_shape))
    model.add(BatchNormalization())   
    model.add(Activation('elu'))            
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))         
    model.add(MaxPooling2D((2,2)))

    l2 = regularizers.l2(l=0.01)

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.7))
    model.add(Dense(2, activation='softmax'))

    return model



def VGG16_Untrained(input_shape):
    model = keras.applications.vgg16.VGG16(
        include_top=True, 
        weights=None, 
        input_shape=input_shape, 
        classes=1,
        input_tensor=None, 
        pooling=None, 
    )

    return model



def VGG16_1(input_shape):
    base_model = keras.applications.VGG16(
        weights='imagenet', 
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    # Freeze Base Layers
    for layer in base_model.layers:
        layer.trainable = False

    # Top Layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.7)(x)
    predictions = Dense(2, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)



def VGG16_2(input_shape):
    base_model = keras.applications.VGG16(
        weights='imagenet', 
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    # Freeze Base Layers
    for layer in base_model.layers:
        layer.trainable = False

    # Top Layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.8)(x)
    x = Dense(256, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.8)(x)
    predictions = Dense(2, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)



def VGG16_Finetune_1(input_shape):
    base_model = keras.applications.VGG16(
        weights='imagenet', 
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    # Freeze Base Layers except topmost block
    base_model.trainable = True
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block5_conv1': set_trainable = True
        layer.trainable = True if set_trainable else False

    # Top Layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.6)(x)
    predictions = Dense(2, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)



def VGG16_Finetune_2(input_shape):
    base_model = keras.applications.VGG16(
        weights='imagenet', 
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    # Freeze Base Layers except topmost block
    base_model.trainable = True
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block4_conv1': set_trainable = True
        layer.trainable = True if set_trainable else False

    # Top Layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.7)(x)
    predictions = Dense(2, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)



def InceptionV4_1(input_shape):
    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
        weights='imagenet', 
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    # Freeze Base Layer
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)

    return Model(input=base_model.input, output=predictions)



def InceptionV4_2(input_shape):
    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
        weights='imagenet', 
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    # Freeze Base Layer
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.6)(x)
    predictions = Dense(2, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


def InceptionV4_3(input_shape):
    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
        weights='imagenet', 
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    # Freeze Base Layer
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.7)(x)
    predictions = Dense(2, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)



def InceptionV4_Finetune_1(input_shape):
    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
        weights='imagenet', 
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    return None
    # Freeze Base Layers except topmost block
    # TODO
    # base_model.trainable = True
    # set_trainable = False
    # for layer in base_model.layers:
    #     if layer.name == 'block5_conv1':
    #         set_trainable = True
    #     layer.trainable = True if set_trainable else False

    # Top Layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.6)(x)
    predictions = Dense(2, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)





# All Implemented models with their associated preprocessors
 
from keras.applications.imagenet_utils import preprocess_input as imagenet_preprocess
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_v2_preprocess
from keras.applications.densenet import preprocess_input as densenet_preprocess

MODELS = [
    (V1_1, imagenet_preprocess),
    (V1_2, imagenet_preprocess),
    (V1_3, imagenet_preprocess),
    (V1_4, imagenet_preprocess),
    (V2, imagenet_preprocess),
    (V3, imagenet_preprocess),
    (V4, imagenet_preprocess),
    (VGG16_Untrained, imagenet_preprocess),
    (VGG16_1, imagenet_preprocess),
    (VGG16_2, imagenet_preprocess),
    (VGG16_Finetune_1, imagenet_preprocess),
    (VGG16_Finetune_2, imagenet_preprocess),
    (InceptionV4_1, imagenet_preprocess),
    (InceptionV4_2, imagenet_preprocess),
    (InceptionV4_3, imagenet_preprocess),
    (InceptionV4_Finetune_1, imagenet_preprocess),
]

def count():
    return len(MODELS)

def get(i, input_shape=None, optimizer=None, metrics=None, loss='categorical_crossentropy'):
    preprocessor = MODELS[i][1]

    if not input_shape or not optimizer or not metrics:
        model = None
    else:
        model = MODELS[i][0](input_shape)
        model.compile(
            loss=loss, 
            optimizer=optimizer, 
            metrics=metrics
        )

    return model, preprocessor
    


if __name__ == "__main__":
    """Executing this script only will output the given model summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, choices=range(0, len(MODELS)), required=True)
    parser.add_argument('--image_size', type=int, nargs='?', default=250)
    args = parser.parse_args()

    model, _ = get(args.model_type, (args.image_size, args.image_size, 3), 'rmsprop', ['acc'])
    model_name = MODELS[args.model_type][0].__name__
    # preprocessor_name = MODELS[args.model_type][1].__name__
    print(f"\nSummarize Model '{model_name}' ({args.model_type})..")
    model.summary()
# -*- coding: utf-8 -*-
"""
Weighted Crossentropy Loss
Reference: https://github.com/keras-team/keras/issues/2115
"""

from keras import backend as K
import numpy as np
import itertools
from sklearn.utils import class_weight



def weights_are_equal(weights, epsilon=0.05):
      return abs(weights[0] - weights[1]) < epsilon


def weights_to_dict(weights):
      return {idx: w for idx, w in enumerate(weights)}


def compute_class_weights(train_gen):
      return class_weight.compute_class_weight('balanced', np.unique(train_gen.classes), train_gen.classes)



class WeightedCategoricalCrossEntropy(object):
    
  def __init__(self, weights):
    nb_cl = len(weights)
    self.weights = np.ones((nb_cl, nb_cl))
    for class_idx, class_weight in weights.items():
      self.weights[0][class_idx] = class_weight
      self.weights[class_idx][0] = class_weight
    self.__name__ = 'w_categorical_crossentropy'

  def __call__(self, y_true, y_pred):
    return self.w_categorical_crossentropy(y_true, y_pred)

  def w_categorical_crossentropy(self, y_true, y_pred):
    nb_cl = len(self.weights)
    final_mask = K.zeros_like(y_pred[..., 0])
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.expand_dims(y_pred_max, axis=-1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        w = K.cast(self.weights[c_t, c_p], K.floatx())
        y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
        y_t = K.cast(y_true[..., c_t], K.floatx())
        final_mask += w * y_p * y_t
    return K.categorical_crossentropy(y_true, y_pred) * final_mask



# # Other Implementation
# def w_categorical_crossentropy(y_true, y_pred, weights):
#     nb_cl = len(weights)
#     final_mask = K.zeros_like(y_pred[:, 0])
#     y_pred_max = K.max(y_pred, axis=1)
#     y_pred_max = K.expand_dims(y_pred_max, 1)
#     y_pred_max_mat = K.equal(y_pred, y_pred_max)
#     for c_p, c_t in product(range(nb_cl), range(nb_cl)):

#         final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
#     return K.categorical_crossentropy(y_pred, y_true) * final_mask
# w_array = np.ones((3,3))
# w_array[2,1] = 1.2
# w_array[1,2] = 1.2
# ncce = partial(w_categorical_crossentropy, weights=w_array)
# ncce.__name__ ='w_categorical_crossentropy'
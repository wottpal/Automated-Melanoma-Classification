# -*- coding: utf-8 -*-
"""Use TensorFlow metrics in Keras."""
"""Source: https://stackoverflow.com/a/50527423/1099817"""

import functools
from keras import backend as K
import tensorflow as tf



def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


auc = as_keras_metric(tf.metrics.auc)
recall = as_keras_metric(tf.metrics.recall)
precision = as_keras_metric(tf.metrics.precision)

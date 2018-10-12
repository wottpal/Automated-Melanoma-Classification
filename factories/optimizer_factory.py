# -*- coding: utf-8 -*-
"""Factory to generate different model optimizers."""

from keras import optimizers



OPTIMIZERS = [

    # Defaults:
    optimizers.RMSprop(lr=1e-3),
    optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
    optimizers.SGD(lr=1e-2, momentum=0.0, decay=0.0, nesterov=False),
    optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),

    # Slower-Variations:
    optimizers.RMSprop(lr=1e-5),
    optimizers.Adam(lr=1e-5),
    optimizers.SGD(lr=1e-5),

    # Others:
    optimizers.RMSprop(lr=2e-5),
    optimizers.RMSprop(lr=1e-5),
    optimizers.SGD(lr=1e-2, momentum=0.0, decay=0.0, nesterov=False),  
    optimizers.SGD(lr=1e-2, momentum=0.9, decay=0.0, nesterov=True),  
    optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/50, nesterov=True),    # Decays over 50 epochs
    optimizers.SGD(lr=1e-4, momentum=0.0, decay=0.0, nesterov=False),  
    optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.0, nesterov=True),  
    optimizers.SGD(lr=1e-5, momentum=0.9, decay=1e-5/20, nesterov=True),    # Decays over 50 epochs

]

def count():
    return len(OPTIMIZERS)

def get(i):
    return OPTIMIZERS[i]
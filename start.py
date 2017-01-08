#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import theano
import numpy as np
import config.config
import train_nnerd

params = config.config.get_nnet_configuration()
def detect():
    x = theano.shared(np.asarray(np.random.RandomState(68).rand(69),
                                 dtype=theano.config.floatX))
    f = theano.function([], theano.tensor.exp(x))
    if not np.any([isinstance(x.op, theano.tensor.Elemwise) and
               ('Gpu' not in type(x.op).__name__)
               for x in f.maker.fgraph.toposort()]):
        print('GPU detected')
        if input('Use it? [n]/y: ').lower().strip() == 'y':
            params['borrow'] = True
            params['processing_unit'] = 'gpu0'
            print('Using the GPU')
            return
    print('Using the CPU')
    return

if __name__ == '__main__':
    print('Initializing')
    detect()
    train_nnerd.train(params)

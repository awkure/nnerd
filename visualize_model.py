# -*- coding: utf-8 -*-
from __future__ import print_function
import six.moves.cPickle as pickle

import os
import sys

import plotly
import plotly.plotly     as pp
import plotly.graph_objs as go

import theano
import theano.tensor as T
from   theano import config

from statsmodels.nonparametric.smoothers_lowess import lowess

import numpy as np

def load():
    path = os.getcwd()
    if not path.endswith('models'):
        path += '/models/'

    if os.path.exists(path):
        files = [[file] for file in os.listdir(path) if file.endswith('.npz')]
        if not files:
            print('There is no *.npz files inside models folder')
            return
        return files, path
    else:
        print('Cannot find models/ folder')
        return

def visualize(obj, filen):
    def line_(d1, d2):
        def make(data, name):
            data.resize(data.shape[0] * data.shape[1] // 2, 2)
            return go.Scattergl(
                x = data[:,0],
                y = data[:,1],
                mode='markers',
                name=name,
                marker = dict(
                    line=dict(
                        width=1)))
        sc1 = make(d1, name="'W'")
        sc2 = make(d2, name="'U'")
        layout = dict(
            legend=dict(
                y=.5,
                traceorder='reversed',
                font=dict(
                    size=16)))
        data = [sc1, sc2]

        return plotly.offline.plot(
            dict(data=data,
                 layout=layout
                ),
            filename='%s_weights.html' % filen)
    def m3dsc(data):
        np.random.shuffle(data)
        return plotly.offline.plot(
            go.Figure(
                data=[go.Surface(z=data)],
                layout=go.Layout(
                    title='History errors (well, train more)',
                    autosize=True)),
            filename='%s_history_errors.html' % filen)

    pt_ = os.getcwd() + '/models/visualization'
    if not os.path.exists(pt_):
        os.mkdir(pt_)
    os.chdir(pt_)
    W = obj['lstm_W']
    U = obj['lstm_U']
    e = obj['history_errs']
    #e = lowess(e[0], e[1], is_sorted=True, frac=0.025, it=0)
    print('Saved. Check the \'%s\' folder' % pt_)
    return line_(W,U), m3dsc(e)

def main():
    files, path = load()
    it_   = 0
    print('Detected files:')
    for file in files:
        print('[%d] %s' % (it_, file[0]))
        it_ += 1
    while True:
        nu = input('Choose the file: ')
        try:
            fl = files[int(nu.strip())]
            print('Choosen: %s' % fl)
            break
        except IndexError:
            print('Type choose the file from 0 to', len(files) - 1)
    obj = np.load(path + fl[0])
    r = visualize(obj, fl[0][:-4])
    return obj

if __name__ == '__main__':
    main()

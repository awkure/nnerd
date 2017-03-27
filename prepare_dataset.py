from __future__ import print_function

import os
import sys
import scipy.io.wavfile
import numpy as np
import operator
import theano
from pipes  import quote
from config import config

params = config.get_nnet_configuration()

class NotImplementedError(Exception):
    pass

def convert_folder(directory):
    if not directory.endswith('/'):
        directory += '/'
    for file in os.listdir(directory):
        if not file.endswith('.wav'):
            to_wav(filename  = directory + file,
                   frequency = params['frequency'])
    return directory + 'wav/'

def to_wav(filename, frequency):
    full_path = [a for a in filename.split('/') if a]
    path      = full_path[0:len(full_path[1:])]
    filename_ = full_path[-1].split('.')[0]
    n_path    = '/' if filename[0] == '/' else ''
    for i in path:
        n_path += i + '/'
    n_path += 'wav/'
    if not os.path.exists(n_path):
        os.mkdir(n_path)

    new_n   = n_path + filename_ + '.wav'
    cmd = 'ffmpeg -loglevel panic -y -i %s -ar %s %s' % \
            (quote(filename),
             frequency,
             quote(new_n))
    os.system(cmd)
    return new_n


def test_sizes(test, data, start = 0):
    if test > 0:
        idx = np.arange(start, len(data[0]))[:test]
        data = (
            [data[0][n] for n in idx \
             if not data[0][n].any == 0],
            [data[1][n] for n in idx \
             if not data[0][n].any == 0]
        )
    return data

def prepare_data(sqs, lbs, mxl=None):
    lgt = [len(s) for s in sqs]
    if mxl is not None:
        sqs_, lbs_, lng_ = [], [], []
        for l, s, y in zip(lgt, sqs, lbs):
            if l < mxl:
                sqs_.append(s)
                lbs_.append(y)
                lng_.append(l)
        lgt = lng_
        lbs = lbs_
        sqs = sqs_
        if len(lgt) < 1:
            return None, None, None
    x_  = np.zeros((np.max(lgt), len(sqs))).astype('int64')
    xm_ = np.zeros((np.max(lgt), len(sqs))).astype(theano.config.floatX)
    for idx, s in enumerate(sqs):
        x_[:lgt[idx], idx]  = s
        xm_[:lgt[idx], idx] = 1.
    return x_, xm_, lbs


def load_data(
    path = params['dataset_dir'],
    positive = False
):
    try:
        path_ = path + 'wav/'
        convert_folder(path)
        if params['mode'].lower() == 'straight':
            r, r_, r2_, r3_ = [], [], [], []
            for file in os.listdir(convert_folder(path)):
                r += [scipy.io.wavfile.read(path_ + file)[1]]

            if params['shuffle_data']:
                np.random.shuffle(r)
            for i in r:
                for j in i:
                    r_ += [j]
            r_  = np.array(r_)

            if params['shuffle_bin_data']:
                np.random.shuffle(r_)

            if positive:
                lz1 = np.where(r_[:,1]>0),1
                lz0 = np.where(r_[:,0]>0),0
                r_[lz0] *= 2
                r_[lz1] *= 2

            for k in np.round(np.abs(r_) / np.max(np.abs(r_))):
                r3_ += [1] if k.any() == 1 else [0]

            data  = ([r_, r3_])
            idxs  = data[0].shape[0]

            train = test_sizes(
                idxs*(2//3),
                data)
            valid = test_sizes(
                idxs//12,
                data,
                start=len(train[0]) // 2)
            test  = test_sizes(
                idxs//4,
                data,
                start=len(train[0]) // 2 + len(valid[0]))

            return  train, valid, test
        else:
            raise NotImplementedError

    except FileNotFoundError:
        print('Cannot find dataset directory. Stopped...')
        return



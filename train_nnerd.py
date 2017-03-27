# -*- coding: utf-8 -*-
from __future__ import print_function
import six.moves.cPickle as pickle

import os
import sys
import time
import scipy.io.wavfile
from   collections import OrderedDict

import theano
import theano.tensor as T
from   theano.sandbox.rng_mrg import MRG_RandomStreams as  RandomStreams
from   theano import config

import numpy as np
import prepare_dataset as pd

from utils.optimizers.optimizers import optimizer
from utils.utils                 import *


params_ = get_nnet_configuration()
THEANO_FLAGS='          \
        floatX=float64, \
        device=%s,      \
        lib.cnmem=1,    \
        exception_verbosity=%s' % \
        (
            params_['processing_unit'],
            params_['exception_verbose']
        )

def train(
    params_,
    patience    = 10,
    disp_freq   = 10,
    decay_c     = 0.,
    batch_size  = 16,
    noise       = 0.,
    mxl         = 100,
    dim_proj    = 128,
    samples     = int(1e+5),
    lrate       = 1e-4,
    encoder     = 'lstm',
    dataset     = 'sound',
    optimizer   = optimizer,
    valid_batch_size = 64

):
    params_['datasets'] = {'sound': (pd.load_data, pd.prepare_data)}
    epochs  = params_['epochs']
    use_dropout = params_['dropout']
    test_size   = params_['test_size']
    valid_size  = params_['valid_size']
    train_size  = params_['train_size']
    reload_model= params_['reload_model']
    sv_freq     = params_['save_frequency']
    vl_freq     = params_['valid_frequency']
    dropout     = params_['dropout']
    saveto      = params_['model_dir'] + params_['model_name']

    model_options = dict(locals().copy(), **params_)
    del model_options['params_']
    print('Model options:')
    mi = np.max(
        np.array(
            [len(x) for x in list(model_options.keys())]
        )
    ) + 2

    for k in sorted(model_options, key=len, reverse=True):
        print('{0:>{1}}: {2}'.format(str(k), str(mi), str(model_options[k])))

    load_data, prepare_data = get_dataset(dataset)
    print('Loading data')
    train, valid, test = pd.load_data(positive = True)
    train = pd.test_sizes(train_size, train)
    valid = pd.test_sizes(valid_size, valid)
    test  = pd.test_sizes(test_size,  test)

    ydim = np.max(train[1]) + 1
    model_options['ydim'] = ydim

    print('Building model')
    params = i_prms(model_options)

    if reload_model:
        l_prms(saveto, params)

    tparams = i_tprms(params)

    (use_noise, x, mask, y,
     f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c       = theano.shared(np_flX(decay_c), name='decay_c', borrow=params['borrow'])
        weight_decay  = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost         += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')
    grads  = T.grad(cost, wrt=list(tparams.values()))
    f_cost = theano.function([x, mask, y],
                             grads,
                             name='f_grad')
    lr = T.scalar(name='lr')
    f_shr, f_upd = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print('Optimization')
    kf_val = mnb_idx(len(valid[0]), valid_batch_size)
    kf_t   = mnb_idx(len(test[0]),  valid_batch_size)
    ll     = len(str(len(train[0])))

    print('{:>{}} train examples'.format(len(train[0]), ll))
    print('{:>{}} valid examples'.format(len(valid[0]), ll))
    print('{:>{}} test  examples'.format(len(test[0]),  ll))

    history_errs = []
    best_p       = None
    bad_count    = 0
    uidx         = 0
    vl_freq = len(train[0]) // batch_size if vl_freq == -1 else vl_freq
    sv_freq = len(train[0]) // batch_size if sv_freq == -1 else sv_freq
    estop = False
    start_time = time.time()
    try:
        for eidx in range(epochs):
            n_smpl = 0

            kf = mnb_idx(
                len(train[0]),
                batch_size,
                shuffle=True
            )

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                x, mask, y = prepare_data(x, y)
                n_smpl += x.shape[1]

                cost = f_shr(x, mask, y)
                f_upd(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    print('Bad cost detected: ', cost)
                    return 1., 1., 1.

                if np.mod(uidx, disp_freq) == 0:
                    print('Epoch', str(eidx)+'/%d'%epochs, '\tUpdate', uidx, '\tCost', cost)

                if saveto and np.mod(uidx, sv_freq) == 0:
                    print('Saving...')
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    saveto = saveto.strip().lower()

                    if False: #os.path.exists(saveto+'.npz'): the fuck wrong with me
                        files = os.listdir(params_['model_dir'])
                        for idx in range(len(files)-1):
                            if files[idx] == params_['model_name']+'.npz' \
                               or type(files[idx][-5]) is not int:
                                del files[idx]
                        nm = np.max(np.array([int(m[-5]) for m in files]))
                        saveto = params_['model_dir'] + params_['model_name'][:-5] + \
                                '_' + str(nm+1) + params_['model_name'][-5:]
                    np.savez(saveto,
                             history_errs = history_errs,
                             **params)
                    pickle.dump(model_options,
                                open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if np.mod(uidx, vl_freq) == 0:
                    print('Validation...')
                    use_noise.set_value(0.)
                    train_err = pr_err(
                        f_pred,
                        prepare_data,
                        train,
                        kf)
                    valid_err = pr_err(
                        f_pred,
                        prepare_data,
                        valid,
                        kf_val)
                    test_err = pr_err(
                        f_pred,
                        prepare_data,
                        test,
                        kf_t)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= np.array(history_errs)[:,0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    print('Train', train_err, '\tValid', valid_err, '\tTest', test_err)

                    if (len(history_errs) > patience and
                        valid_err >= np.array(history_errs)[:-patience,0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early stop')
                            estop = True
                            break
            print('Seen %d samples' % n_smpl)

            if estop:
                break

    except KeyboardInterrupt:
        print('Training interrupted')
        print('Saving...')

    end_time = time.time()
    if best_p is not None:
        zip_(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = mnb_idx(len(train[0]), batch_size)
    train_err = pr_err(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pr_err(f_pred, prepare_data, valid, kf_val)
    test_err  = pr_err(f_pred, prepare_data, test,  kf_t)

    print('\tTrain', train_err, '\tValid', valid_err, '\tTest', test_err)
    if saveto:
        np.savez(saveto, train_err = train_err,
                 valid_err = valid_err, test_err = test_err,
                 history_errs = history_errs, **best_p)

    print('Saved to: \'%s\'' % (os.getcwd() + '/' + saveto))
    print('The code run for %d epochs, with %d sec/epochs' %
          ((eidx + 1), int((end_time - start_time) / (float(eidx + 1)))))
    print( ('Training took %.1fs' % (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err

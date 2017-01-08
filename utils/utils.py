# -*- coding: utf-8 -*-
from  __future__ import print_function
import six.moves.cPickle as pickle

import scipy.io.wavfile
from   itertools   import *
from   collections import OrderedDict

import theano
import theano.tensor as T
from   theano.sandbox.rng_mrg import MRG_RandomStreams as  RandomStreams
from   theano import config

import numpy as np
import prepare_dataset as pd

from config.config import get_nnet_configuration


params = get_nnet_configuration()
params['datasets'] = {
    'sound': (pd.load_data, pd.prepare_data)
}
np.random.seed(params['random_seed'])


def mnb_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype='int32')

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches =[]
    minibatch_start = 0

    for i in range(n // minibatch_size):
        minibatches.append(
            idx_list[minibatch_start : minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def get_dataset(name):
    return (
        params['datasets'][name][0],
        params['datasets'][name][1]
    )

def zip_(params, tparams):
    for k, v in params.items():
        tparams[k].set_value(v)

def unzip(z):
    n = OrderedDict()
    for k, v in z.items():
        n[k] = v.get_value()
    return n

def dr_lr(s_b, noise, trng):
    return  T.switch(noise,
                     (s_b* trng.binomial
                      (s_b.shape,
                       p = .5,
                       n = 1,
                       dtype = s_b.dtype)),
                     s_b * .5)

def _p(p, name):
    return '%s_%s' % (p,name)

def np_flX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def i_prms(opts):
    prms = OrderedDict()
    randn = np.random.rand(opts['samples'],
                           opts['dim_proj'])
    prms['Wemb'] = (.01 * randn).astype(config.floatX)
    prms = get_layer(opts['encoder'])[0](opts,prms,prefix=opts['encoder'])
    prms['U'] = .01 * np.random.randn(
        opts['dim_proj'],
        opts['ydim']).astype(config.floatX)
    prms['b'] = np.zeros((opts['ydim'],)).astype(config.floatX)
    return prms

def l_prms(path, prms):
    p = np.load(path)
    for k, v in prms.items():
        if k not in p:
            raise Warning('%s is not in the archive' % k)
        prms[k] = p[k]

    return prms

def i_tprms(p_):
    tp = OrderedDict()
    for k, p in p_.items():
        tp[k] = theano.shared(p_[k], name=k, borrow=params['borrow'])
    return tp

def get_layer(n):
    return layers[n]

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u = np.linalg.svd(W)[0]
    return u.astype(config.floatX)

def prm_i_lstm(opts, prms, prefix='lstm'):
    W = np.concatenate([ortho_weight(opts['dim_proj'])] * 4, axis=1)
    U = np.concatenate([ortho_weight(opts['dim_proj'])] * 4, axis=1)
    b = np.zeros((4 * opts['dim_proj'],))
    prms[_p(prefix, 'W')] = W
    prms[_p(prefix, 'U')] = U
    prms[_p(prefix, 'b')] = b.astype(config.floatX)
    return prms

def layer(tprms,
          st_bl,
          opts,
          prefix='lstm',
          mask=None):
    nst = st_bl.shape[0]
    nsm = st_bl.shape[1] \
            if st_bl.ndim == 3 else 1

    assert mask is not None, 'mask inside layer is None'

    def _step(m_, x_, h_, c_):
        prct  = T.dot(h_, tprms[_p(prefix, 'U')])
        prct += x_
        def ts(n):
            return T.nnet.sigmoid\
                    (_slice(prct, n, opts['dim_proj']))
        i = ts(0)
        f = ts(1)
        o = ts(2)
        c = T.tanh(_slice(prct, 3, opts['dim_proj']))
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h, c

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[...,n*dim:(n+1)*dim]
        return _x[:,n*dim:(n+1)*dim]

    st_bl = (T.dot(st_bl, tprms[_p(prefix, 'W')]) +
                   tprms[_p(prefix, 'b')])
    rval, updates = theano.scan(
        _step,
        sequences=[mask, st_bl],
        outputs_info=[T.alloc(np_flX(0.),
                              nsm,
                              opts['dim_proj']),
                      T.alloc(np_flX(0.),
                              nsm,
                              opts['dim_proj'])],
        name=_p(prefix, '_layers'),
        n_steps = nst)
    return rval[0]

layers = {'lstm': (prm_i_lstm, layer)}

def softmax(x):
    return T.concatenate(
        [
            T.exp(x)[:,:3] / T.exp(x)[:,:3].sum(axis=1)[:,None],
            T.exp(x)[:,3:] / T.exp(x)[:,3:].sum(axis=1)[:,None]
        ], axis=1
    )

def build_model(tprms, opts):
    trng = RandomStreams(params['random_seed'])
    noise = theano.shared(np_flX(0.), borrow=params['borrow'])

    x    = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=config.floatX)
    y    = T.vector('y', dtype='int64')

    n_tm = x.shape[0]
    n_sm = x.shape[1]
    emb = tprms['Wemb'][x.flatten()].reshape([n_tm,
                                              n_sm,
                                              opts['dim_proj']])
    proj = get_layer(opts['encoder'])[1](tprms, emb, opts,
                                            prefix=opts['encoder'],
                                            mask=mask)
    if opts['encoder'].lower() == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if opts['use_dropout']:
        proj = dr_lr(proj, noise, trng)
   #pred = T.nnet.softmax(T.dot(proj, tprms['U']) + tprms['b'])
    pred = softmax(T.dot(proj, tprms['U']) + tprms['b'])
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')
    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    cost = -T.log(pred[T.arange(n_sm), y] + off).mean()
    return noise, x, mask, y, f_pred_prob, f_pred, cost

def pr_prob(f_pp, prepare_data, data, it, verbose=True):
    p_   = np.zeros((len(data[0]), 2)).astype(config.floatX)
    done = 0
    for _, v_idx in it:
        x, mask, y = prepare_data([data[0][t] for t in v_idx],
                                  np.array(data[1])[v_idx],
                                  mxl=None)
        pred_p_ = f_pp(x, mask)
        p_[v_idx,:] = pr_prob
        done += len(v_idx)
        if verbose:
            print('%d/%d samples classified' % (done, len(data[0])))
    return p_

def pr_err(fp, p_data, data, it, verbose=False):
    valid_err = 0
    for _, valid_index in it:
        x, mask, y = p_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index],
                                  mxl=None)
        valid_err += (fp(x, mask) == np.array(data[1])[valid_index]).sum()
    valid_err = 1. - np_flX(valid_err) / len(data[0])
    return valid_err


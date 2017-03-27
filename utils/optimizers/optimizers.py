import numpy as np
import theano
import theano.tensor as T
from   theano import config

def optimizer(lr, tparams, grads, x, mask, y, cost):
    def mki(mes):
        natkf = np.asarray(0., dtype=theano.config.floatX)
        return [theano.shared(p.get_value() * natkf, name=mes % k) for k, p in tparams.items()]

    zgr   = mki('%s_grad')
    rp2   = mki('%s_rup2')
    rgr2  = mki('%s_rgrad2')
    zgup  = [(zg, g) for zg, g in zip(zgr, grads)]
    rg2up = [(rg2, .95 * rg2 + .05 * (g ** 2)) for rg2, g in zip(rgr2, grads)]
    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zgr,rp2,rgr2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(rp2, updir)]
    p_up  = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
    f_shr = theano.function([x, mask, y], cost, updates=zgup + rg2up, name='opt_f_shr')
    f_upd = theano.function([lr], [], updates=ru2up + p_up, on_unused_input='ignore',name='opt_f_upd')

    return f_shr, f_upd



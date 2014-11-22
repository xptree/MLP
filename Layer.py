#!/usr/bin/env python
# encoding: utf-8
# File Name: HiddenLayer.py
# Author: Jiezhong Qiu
# Create Time: 2014/11/22 13:46
# TODO: implemetn HiddenLayer

import numpy as np
from ActivationFunction import sigmoid

class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None,
            nxt=None, Lambda = 0., alpha = 0.001):
        if W is None:
            W = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6. / (n_in + n_out)),
                    high = np.sqrt(6. / (n_in + n_out)),
                    size = (n_out, n_in)
                ),
                dtype = np.float
            )
        if b is None:
            b = np.zeros(n_out, dtype=np.float).reshape(n_out, 1)
        self.W, self.b = W * 4, b
        self.n_in, self.n_out = n_in, n_out
        self.nxt = nxt
        self.Lambda = Lambda #weight decay
        self.alpha = alpha #learning rate
    def process(self, a_in, m):
        z = np.dot(self.W, a_in) + np.tile(self.b, (1, m))
        a_out = self.activation.f(z)
        delta = self.nxt.process(a_out, m)
        gradW = np.dot(delta, a_in.T) + self.Lambda * self.W
        gradb = np.dot(delta, np.ones((m, 1)))
        self.W -= self.alpha * gradW
        self.b -= self.alpha * gradb
        return np.dot( np.dot(a_in, 1 - a_in), np.dot(self.W.T, delta) )




if __name__ == "__main__":
    pass



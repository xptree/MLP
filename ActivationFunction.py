#!/usr/bin/env python
# encoding: utf-8
# File Name: ActivationFunction.py
# Author: Jiezhong Qiu
# Create Time: 2014/11/22 13:49
# TODO: implement activation functions

import numpy as np

class sigmoid(object):
    @staticmethod
    def f(z):
        return 1. / (1 + np.exp(z))
    @staticmethod
    def fprime(z):
        g = sigmoid.f(z)
        return g * (1 - g)

if __name__ == "__main__":
    z = np.array([-10, -5, 0, 5, 10], dtype = np.float)
    print sigmoid.f(z)
    print sigmoid.fprime(z)


#!/usr/bin/env python
# encoding: utf-8
# File Name: MLP.py
# Author: Jiezhong Qiu
# Create Time: 2014/11/22 13:20
# TODO: implement MLP

import numpy as np
from Layer import InputLayer
from Layer import OutputLayer
from Layer import HiddenLayer

class MLP(object):
    def __init__(self, config, Lambda = .001, alpha = .001, activation="sigmoid", costType="mse"):
        if activation == "sigmoid":
            activation = lambda x: 1. / (1 + np.exp(x))
            activationPrime = lambda x : x * ( 1. - x )
        elif activation == "tanh":
            activation = np.tanh
            activationPrime = lambda x: 1. - x ** 2
        rng = np.random.RandomState(1234)
        self.layers = []
        for i in range(len(config))[::-1]:
            if i == len(config) - 1:
                #output
               self.layers.append(OutputLayer(config[i], activation, activationPrime, costType))
            elif i == 0:
                self.layers.append(InputLayer(rng, config[i], config[i + 1], activation,
                    activationPrime, self.layers[-1], Lambda, alpha))
            else:
                self.layers.append(HiddenLayer(rng, config[i], config[i + 1], activation,
                    activationPrime, self.layers[-1], Lambda, alpha))
    def fit(self, X_train, y_train, iteration, batch_size):
        M = X_train.shape[1]
        batch_n = M / batch_size
        if batch_n * batch_size < M:
            batch_n += 1
        for i in xrange(iteration):
           #random shuffle
           for batch in xrange(batch_n):
               start = batch * batch_size
               end = min( (batch + 1) * batch_size, M )
               self.train(X_train[:, [start, end]], y_train[:, [start, end]])
    def predict(self, X_test):
        return self.predict(X_test)

    def train(self, X_train, y_train):
        inputlayer = self.layers[-1]
        outputlayer = self.layers[0]
        outputlayer.setstd(y_train)
        inputlayer.process(X_train,  bp = True)
        cost = sum([layer.cost for layer in self.layers])
        print cost
    def test(self, X_test):
        inputlayer = self.layers[-1]
        inputlayer.process(X_test, bp = False)
        outputlayer = self.layers[0]
        return outputlayer.getresult()

if __name__ == "__main__":
    pass



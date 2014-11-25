#!/usr/bin/env python
# encoding: utf-8
# File Name: MLP.py
# Author: Jiezhong Qiu
# Create Time: 2014/11/22 13:20
# TODO: implement MLP

import numpy as np
import logging
from Layer import InputLayer
from Layer import OutputLayer
from Layer import HiddenLayer
from Layer import SoftmaxLayer

class MLP(object):
    def __init__(self, config, Lambda = .001, alpha = .01, activation="sigmoid", costType="mse", load=False):
        if activation == "sigmoid":
            from scipy.special import expit
            activation = expit
            activationPrime = lambda x : x * ( 1. - x )
        elif activation == "tanh":
            activation = np.tanh
            activationPrime = lambda x: 1. - x ** 2

        if costType == "mse":
            error = lambda std_outputs, outputs: \
                    np.multiply(outputs - std_outputs, activationPrime(outputs))
            costFunc = lambda std_outputs, outputs: \
                    0.5 * np.sum((std_outputs-outputs)**2)
        elif costType == "softmax":
            error = lambda std_outputs, outputs: \
                    (outputs - std_outputs)
            costFunc  = lambda std_outputs, outputs: \
                    -(np.sum(np.multiply(std_outputs, np.log(outputs))))
        rng = np.random.RandomState(1234)
        self.layers = []
        for i in range(len(config))[::-1]:
            if i == len(config) - 1:
                #output
               self.layers.append(OutputLayer(config[i],  activation, activationPrime, error, costFunc))
            elif i == 0:
                self.layers.append(InputLayer(rng, config[i], config[i + 1], activation,
                    activationPrime, self.layers[-1], Lambda, alpha))
            else:
                layer = SoftmaxLayer(rng, config[i], config[i + 1],
                        activation, activationPrime, self.layers[-1], Lambda, alpha) if i==1 and costType == "softmax" else HiddenLayer(rng, config[i], config[i+1],
                        activation, activationPrime, self.layers[-1], Lambda, alpha)
                self.layers.append(layer)
                #self.layers.append(HiddenLayer(rng, config[i], config[i + 1], activation,
                #    activationPrime, self.layers[-1], Lambda, alpha))
    def fit(self, X_train, y_train, iteration, batch_size):
        M = X_train.shape[1]
        batch_n = M / batch_size
        if batch_n * batch_size < M:
            batch_n += 1
        for i in xrange(iteration):
            logging.info("Training #iteration = %d" % i)
           #random shuffle
            for batch in xrange(batch_n):
                start = batch * batch_size
                end = min( (batch + 1) * batch_size, M )
                self.train(X_train[:, start:end], y_train[:, start:end], log = (batch == 0))

    def train(self, X_train, y_train, log = False):
        inputlayer = self.layers[-1]
        outputlayer = self.layers[0]
        outputlayer.setstd(y_train)
        inputlayer.process(X_train,  bp = True)
        cost = sum([layer.cost for layer in self.layers])
        if log:
            logging.info("Cost = %f" % cost)
    def predict(self, X_test, y_test):
        inputlayer = self.layers[-1]
        inputlayer.process(X_test, bp = False)
        outputlayer = self.layers[0]
        y_res = outputlayer.getresult()
        y_pred = np.argmax(y_res, axis=0)
        y_std = np.argmax(y_test, axis=0)
        from sklearn.metrics import confusion_matrix, accuracy_score
        print accuracy_score(y_std, y_pred)
        print confusion_matrix(y_std, y_pred)
        return y_res, y_pred
if __name__ == "__main__":
    pass



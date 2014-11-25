#!/usr/bin/env python
# encoding: utf-8
# File Name: MNIST.py
# Author: Jiezhong Qiu
# Create Time: 2014/11/24 15:53
# TODO:

import cPickle, gzip
import numpy as np
from MLP import MLP
import unittest
import util


import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

def translate(dataset):
    X = dataset[0].T.astype(util.FLOAT, copy=False)
    y = []
    for inst in dataset[1]:
        y.append(np.eye(10)[inst])
    y = np.array(y, dtype = util.FLOAT).T
    return X, y

class mnistTestCase(unittest.TestCase):
    def runTest(self):
        MNIST_DIR = "../Data/mnist.pkl.gz"
        f = gzip.open(MNIST_DIR, "rb")
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        X_train, y_train = translate(train_set)
        X_test, y_test = translate(test_set)
        mlp = MLP([784, 800, 300, 10], 0, 0.2, "sigmoid", "mse", load="True")
        mlp.fit(X_train, y_train, 400, 500000)
        mlp_result, mlp_prediction = mlp.predict(X_test, y_test)
        mlp_result, mlp_prediction = mlp.predict(X_train, y_train)
        loss = np.mean( (mlp_result - y_train)**2)
        print "Loss: ", loss
        error = sum([mlp_prediction[i] != train_set[1][i] for i in xrange(len(mlp_prediction))])
        error /= float(len(mlp_prediction))
        print "Error: ", error
        self.assertTrue(error < .1)
if __name__ == "__main__":
    unittest.main()

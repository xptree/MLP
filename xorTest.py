#!/usr/bin/env python
# encoding: utf-8
# File Name: xorTest.py
# Author: Jiezhong Qiu
# Create Time: 2014/11/24 16:57
# TODO:

import util
import unittest
import numpy as np
from MLP import MLP

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

class XORTestCase(unittest.TestCase):
    def runTest(self):
        inputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype = util.FLOAT).T
        outputs = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype = util.FLOAT).T
        mlp = MLP([2, 5, 2], Lambda = 0.0001, alpha = .1, activation = "sigmoid", costType = "mse")
        mlp.fit(inputs, outputs, 20000, 4)
        mlp_result, mlp.prediction = mlp.predict(inputs, outputs)
        loss = np.mean( (mlp_result - outputs)**2)
        print "Prediction: ", mlp_result
        print "Loss: ", loss
        np.testing.assert_almost_equal(loss, 0, 2)

if __name__ == "__main__":
    unittest.main()

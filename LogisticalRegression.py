import numpy as np
import pandas as pd
import random as rand


class LogisticalRegression:
    # know it says task1 but this is abstract logistical regression
    # X is the input space
    # X has to be a pandas dataframe of the input matrix
    # Y has to be a pandas dataframe of the input matrix
    def __init__(self, X, Y, learningRate, iterations):
        self.X = X
        self.Y = Y
        self.learningRate = learningRate
        self.iterations = iterations

    # trains the data and once the training is done, it returns W and B, which can be used
    # to predict new images
    def train(self):
        m = self.X.shape[0]
        n = self.X.shape[1]
        W = np.zeros(n)
        for _ in range(1):
            rand_data_idx = rand.randint(0, m - 1)
            Z = np.dot(W.T, self.X[rand_data_idx])
            print(Z)
            # sigmoid function
            A = 1 / (1 + np.exp(-1 * Z))
            print(A - self.Y[rand_data_idx])
            print(A - self.Y[rand_data_idx])
            print((A - self.Y[rand_data_idx]) * self.X[rand_data_idx])
            W = W - (self.learningRate * (A - self.Y[rand_data_idx]) * self.X[rand_data_idx])
        return W

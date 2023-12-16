import numpy as np
import random as rand
import matplotlib.pyplot as plt


class LogisticalRegression:
    def __init__(self, X, Y, learningRate, iterations):
        self.X = X
        self.Y = Y
        self.learningRate = learningRate
        self.iterations = iterations
        self.epochs = []
        self.loss_values = []
        self.weights = self.train()

    # trains the data and once the training is done, it returns W and B, which can be used
    # to predict new images
    def train(self):
        m = self.X.shape[0]
        n = self.X.shape[1]
        W = np.zeros(n)
        for epoch in range(self.iterations):
            rand_data_idx = epoch
            Z = np.dot(W.T, self.X[rand_data_idx])
            # sigmoid function
            A = self.sigmoid(Z)
            loss = 0
            for i in range(m):
                loss += (-1 * self.Y[i] * np.log(A)) - ((1 - self.Y[i]) * np.log(1 - A))
            loss /= m
            self.epochs.append(epoch)
            self.loss_values.append(loss)
            W = W - (
                self.learningRate * (A - self.Y[rand_data_idx]) * self.X[rand_data_idx]
            )
        return W

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def predict(self, X):
        confidence = np.dot(self.weights.T, X)
        return 0 if confidence < 50 else 1

    def graph_loss_over_time(self):
        plt.plot(self.epochs, self.loss_values, label="Training Loss")
        plt.title("Loss Over Time")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    # early termination
    # look over loss data and save a few of the past weights so that we can revert to an earlier model

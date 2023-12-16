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

    def train(self):
        m = self.X.shape[0]
        n = self.X.shape[1]
        W = np.random.uniform(0, 0, n)
        for epoch in range(self.iterations):
            loss = 0
            cnt = 0
            for _ in range(m):
                rand_data_idx = rand.randint(0, m - 1)
                Z = np.dot(W, self.X[rand_data_idx])
                # sigmoid function
                A = self.sigmoid(Z)
                prediction = self.predict(W, self.X[rand_data_idx])
                cnt += 1 if prediction == self.Y[rand_data_idx] else 0
                loss += (-1 * self.Y[rand_data_idx] * np.log(A)) - (
                    (1 - self.Y[rand_data_idx]) * np.log(1 - A)
                )
                W = W - (
                    self.learningRate
                    * (A - self.Y[rand_data_idx])
                    * self.X[rand_data_idx]
                )
            cnt /= m
            loss /= m
            self.epochs.append(epoch)
            self.loss_values.append(loss)
            print("At epoch", epoch, "loss =", loss, "avg prediction =", cnt)
        return W

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def predict(self, W, X):
        Z = np.dot(W, X)
        A = self.sigmoid(Z)
        return 0 if A < 0.5 else 1

    def graph_loss_over_time(self):
        plt.plot(self.epochs, self.loss_values, label="Training Loss")
        plt.title("Loss Over Time")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def loss(self, W, X, Y):
        Z = np.dot(W, X)
        A = self.sigmoid(Z)
        return (-1 * Y * np.log(A)) - ((1 - Y) * np.log(1 - A))

    # early termination
    # look over loss data and save a few of the past weights so that we can revert to an earlier model

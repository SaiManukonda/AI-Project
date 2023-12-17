import numpy as np
import random as rand
import matplotlib.pyplot as plt


class LogisticalRegression:
    def __init__(self, X, Y, learningRate, iterations, X_v, Y_v):
        self.X = X
        self.Y = Y
        self.X_v = X_v
        self.Y_v = Y_v
        self.learningRate = learningRate
        self.iterations = iterations
        self.epochs = []
        self.loss_values = []
        self.v_loss_values = []
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
            v_loss = 0
            for i in range(len(self.X_v)):
                Z = np.dot(W, self.X_v[i])
                A = self.sigmoid(Z)
                v_loss += (-1 * self.Y_v[i] * np.log(A)) - (
                    (1 - self.Y_v[i]) * np.log(1 - A)
                )
            v_loss /= len(self.X_v)
            self.v_loss_values.append(v_loss)
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
        plt.plot(self.epochs, self.v_loss_values, label="Validation Loss")
        plt.title("Loss Over Time 5000 examples")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

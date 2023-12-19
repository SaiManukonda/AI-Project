import numpy as np
import random as rand
import matplotlib.pyplot as plt


class SoftmaxRegression:
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
        weights = []
        for _ in range(4):
            weights.append(np.random.uniform(0, 0, n))
        for epoch in range(self.iterations):
            loss = np.zeros(4)
            cnt = 0
            for _ in range(m):
                rand_data_idx = rand.randint(0, m - 1)
                A = []
                for j in range(self.Y.shape[1]):
                    Z = np.dot(weights[j], self.X[rand_data_idx])
                    A.append(np.exp(Z))
                A /= sum(A)
                cnt += 1 if np.argmax(A) == np.argmax(self.Y[rand_data_idx]) else 0
                loss += -1 * self.Y[rand_data_idx] * np.log(A)
                for j in range(4):
                    weights[j] = weights[j] - (
                        self.learningRate
                        * (A[j] - self.Y[rand_data_idx][j])
                        * self.X[rand_data_idx]
                    )
            cnt /= m
            loss /= m
            v_loss = np.zeros(4)
            for i in range(self.X_v.shape[0]):
                A = []
                for j in range(self.Y_v.shape[1]):
                    Z = np.dot(weights[j], self.X_v[i])
                    A.append(np.exp(Z))
                A /= sum(A)
                v_loss += -1 * self.Y_v[i] * np.log(A)
            v_loss /= self.X_v.shape[0]
            self.v_loss_values.append(sum(v_loss))
            self.epochs.append(epoch)
            self.loss_values.append(sum(loss))
            print(
                "At epoch", epoch, "loss =", loss, "avg prediction success rate =", cnt
            )
        return weights

    def predict(self, W, X):
        A = []
        for i in range(self.Y.shape[1]):
            Z = np.dot(W[i], X)
            A.append(np.exp(Z))
        A /= sum(A)
        return np.argmax(A)

    def graph_loss_over_time(self):
        plt.plot(self.epochs, self.loss_values, label="Training Loss")
        plt.plot(self.epochs, self.v_loss_values, label="Validation Loss")
        plt.title("Sum of Loss Over each Category Over Time 5000 Examples")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

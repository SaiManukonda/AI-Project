import numpy as np
import pandas as pd

class Task1:
    #know it says task1 but this is abstract logistical regression 
    #X is the input space
    #X has to be a pandas dataframe of the input matrix
    #Y has to be a pandas dataframe of the input matrix
    def __init__(self, X, Y, learningRate, iterations):
        self.X = X
        self.Y = Y
        self.learningRate = learningRate
        self.iterations = iterations
    
    #trains the data and once the training is done, it returns W and B, which can be used
    #to predict new images
    def train(self):
        m = self.X.shape[1]
        n = self.Y.shape[0]
        
        W = np.zeroes((n, 1))
        B = 0
        
        for i in range(self.iterations):
            Z = np.dot(W.T, self.X) + B
            #sigmoid function
            A = 1/(1+np.exp(-Z))
            
            dw = (1/m)*np.dot(A-Y, X.t)
            db = (1/m)*np.sum(A - Y)
            
            W = W - self.learningRate * dw.T
            B = B = self.learningRate * db
        
        return W, B
        
            
        
        
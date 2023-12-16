import numpy as np
import pandas as pd

class LogisticalRegression:
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
        n = self.X.shape[0]
        
        W = np.zeros((n, 1))
        B = 0
        print(W.shape)
        
        for i in range(self.iterations):
            print(W.T.shape)
            print(self.X.shape)
            Z = np.dot(W.T, self.X) + B
            print("1")
            #sigmoid function
            A = 1/(1+np.exp(-1*Z))
            
            cost = -(1/m)*np.sum(self.Y*np.log(A) + (1-self.Y)*np.log(1-A))
            
            dw = (1/m)*np.dot(A-self.Y, self.X.T)
            db = (1/m)*np.sum(A - self.Y)
            
            W = W - self.learningRate * dw.T
            B = B - self.learningRate * db
            
            if(i%(self.iterations/10) == 0):
                print("Cost = " + str(cost))

        return W, B
        
            
        
        
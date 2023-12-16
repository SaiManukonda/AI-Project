from WiringDiagram import WiringDiagram
import numpy as np
import pandas as pd
from LogisticalRegression import LogisticalRegression

# creating a data set of 1000 inputs and outputs
X = []
Y = []
for i in range(1000):
    currentWiring = WiringDiagram()
    flattenedFeatureArray = []
    # matrix of vector representation of the colors
    mat = []
    for i in range(len(currentWiring.diagram)):
        for f in range(len(currentWiring.diagram)):
            if currentWiring.diagram[i, f] == 0:
                mat.append([0, 0, 0, 0])
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(0)
            elif currentWiring.diagram[i, f] == 1:
                mat.append([1, 0, 0, 0])
                flattenedFeatureArray.append(1)
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(0)
            elif currentWiring.diagram[i, f] == 2:
                mat.append([0, 1, 0, 0])
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(1)
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(0)
            elif currentWiring.diagram[i, f] == 3:
                mat.append([0, 0, 1, 0])
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(1)
                flattenedFeatureArray.append(0)
            else:
                mat.append([0, 0, 0, 1])
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(0)
                flattenedFeatureArray.append(1)
    for i in range(len(mat)):
        if i % 2 == 0:
            firstMatrix = np.array(mat[i])
            secondMatrix = np.array(mat[i + 1])
            flattenedFeatureArray.append(np.dot(firstMatrix, secondMatrix))
    mat = []
    for i in range(len(currentWiring.diagram)):
        for f in range(len(currentWiring.diagram)):
            if currentWiring.diagram[f, i] == 0:
                mat.append([0, 0, 0, 0])
            elif currentWiring.diagram[f, i] == 1:
                mat.append([1, 0, 0, 0])
            elif currentWiring.diagram[f, i] == 2:
                mat.append([0, 1, 0, 0])
            elif currentWiring.diagram[f, i] == 3:
                mat.append([0, 0, 1, 0])
            else:
                mat.append([0, 0, 0, 1])
    for i in range(len(mat)):
        if i % 2 == 0:
            firstMatrix = np.array(mat[i])
            secondMatrix = np.array(mat[i + 1])
            flattenedFeatureArray.append(np.dot(firstMatrix, secondMatrix))
    X.append(flattenedFeatureArray)
    Y.append(currentWiring.dangerous)
X = np.array(X)
Y = np.array(Y)
X_Train = X
Y_Train = Y

Train_1000 = LogisticalRegression(X_Train, Y_Train, 0.0005, 10000)
print(Train_1000.train())

from WiringDiagram import WiringDiagram
import numpy as np
from LogisticalRegression import LogisticalRegression


# loops over the wire diagram and creates one hot encoded array of features
def extractFeatures(current_wiring):
    one_hot = []
    # count of each color
    vect = [0 for _ in range(4)]
    for i in range(0, len(current_wiring)):
        for j in range(0, len(current_wiring)):
            if current_wiring[i][j] == 1:
                vect[0] += 1
            elif current_wiring[i][j] == 2:
                vect[1] += 1
            elif current_wiring[i][j] == 3:
                vect[2] += 1
            elif current_wiring[i][j] == 4:
                vect[3] += 1
    for k in range(4):
        one_hot.append(vect[k])
        
    # for i in range(len(current_wiring)):
    #     for j in range(len(current_wiring)):
    #         vect = [0 for _ in range(4)]
    #         if current_wiring[i][j] == 1:
    #             vect[0] += 1
    #         elif current_wiring[i][j] == 2:
    #             vect[1] += 1
    #         elif current_wiring[i][j] == 3:
    #             vect[2] += 1
    #         elif current_wiring[i][j] == 4:
    #             vect[3] += 1
    #         for k in range(4):
    #             one_hot.append(vect[k])
          
    # loop over each two by two patch and average pool
    # for i in range(0, len(current_wiring), 2):
    #     for j in range(0, len(current_wiring), 2):
    #         vect = [0 for _ in range(4)]
    #         for k in range(2):
    #             for l in range(2):
    #                 if current_wiring[i + k][j + l] == 1:
    #                     vect[0] += 1
    #                 elif current_wiring[i + k][j + l] == 2:
    #                     vect[1] += 1
    #                 elif current_wiring[i + k][j + l] == 3:
    #                     vect[2] += 1
    #                 elif current_wiring[i + k][j + l] == 4:
    #                     vect[3] += 1
    #         for m in range(4):
    #             one_hot.append(vect[m] / 4)
    # loop over each two by one patch and count number if each kind of transition
    vect = [0 for _ in range(12)]
    for l in range(3):
        for i in range(l, len(current_wiring) - 2, 3):
            for j in range(len(current_wiring)):
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i + 1][j] == 2
                    and current_wiring[i + 2][j] == 1
                ):
                    vect[0] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i + 1][j] == 1
                    and current_wiring[i + 2][j] == 2
                ):
                    vect[1] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i + 1][j] == 3
                    and current_wiring[i + 2][j] == 2
                ):
                    vect[2] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 2
                    and current_wiring[i + 2][j] == 3
                ):
                    vect[3] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 4
                    and current_wiring[i + 2][j] == 3
                ):
                    vect[4] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i + 1][j] == 3
                    and current_wiring[i + 2][j] == 4
                ):
                    vect[5] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i + 1][j] == 3
                    and current_wiring[i + 2][j] == 1
                ):
                    vect[6] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 1
                    and current_wiring[i + 2][j] == 3
                ):
                    vect[7] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i + 1][j] == 4
                    and current_wiring[i + 2][j] == 1
                ):
                    vect[8] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i + 1][j] == 1
                    and current_wiring[i + 2][j] == 4
                ):
                    vect[9] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i + 1][j] == 2
                    and current_wiring[i + 2][j] == 4
                ):
                    vect[10] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i + 1][j] == 4
                    and current_wiring[i + 2][j] == 2
                ):
                    vect[11] += 1
    for l in range(3):
        for i in range(len(current_wiring)):
            for j in range(l, len(current_wiring) - 2, 3):
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i][j + 1] == 2
                    and current_wiring[i][j + 2] == 1
                ):
                    vect[0] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i][j + 1] == 1
                    and current_wiring[i][j + 2] == 2
                ):
                    vect[1] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i][j + 1] == 3
                    and current_wiring[i][j + 2] == 2
                ):
                    vect[2] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 2
                    and current_wiring[i][j + 2] == 3
                ):
                    vect[3] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 4
                    and current_wiring[i][j + 2] == 3
                ):
                    vect[4] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i][j + 1] == 3
                    and current_wiring[i][j + 2] == 4
                ):
                    vect[5] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i][j + 1] == 3
                    and current_wiring[i][j + 2] == 1
                ):
                    vect[6] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 1
                    and current_wiring[i][j + 2] == 3
                ):
                    vect[7] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i][j + 1] == 4
                    and current_wiring[i][j + 2] == 1
                ):
                    vect[8] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i][j + 1] == 1
                    and current_wiring[i][j + 2] == 4
                ):
                    vect[9] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i][j + 1] == 2
                    and current_wiring[i][j + 2] == 4
                ):
                    vect[10] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i][j + 1] == 4
                    and current_wiring[i][j + 2] == 2
                ):
                    vect[11] += 1
    for k in range(12):
        one_hot.append(vect[k])
    vect = [0 for _ in range(24)]
    for l in range(4):
        for i in range(l, len(current_wiring) - 3, 4):
            for j in range(len(current_wiring)):
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i + 1][j] == 2
                    and current_wiring[i + 2][j] == 3
                    and current_wiring[i + 3][j] == 1
                ):
                    vect[0] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i + 1][j] == 3
                    and current_wiring[i + 2][j] == 2
                    and current_wiring[i + 3][j] == 1
                ):
                    vect[1] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i + 1][j] == 2
                    and current_wiring[i + 2][j] == 4
                    and current_wiring[i + 3][j] == 1
                ):
                    vect[2] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i + 1][j] == 4
                    and current_wiring[i + 2][j] == 2
                    and current_wiring[i + 3][j] == 1
                ):
                    vect[3] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i + 1][j] == 3
                    and current_wiring[i + 2][j] == 4
                    and current_wiring[i + 3][j] == 1
                ):
                    vect[4] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i + 1][j] == 4
                    and current_wiring[i + 2][j] == 3
                    and current_wiring[i + 3][j] == 1
                ):
                    vect[5] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i + 1][j] == 1
                    and current_wiring[i + 2][j] == 3
                    and current_wiring[i + 3][j] == 2
                ):
                    vect[6] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i + 1][j] == 3
                    and current_wiring[i + 2][j] == 1
                    and current_wiring[i + 3][j] == 2
                ):
                    vect[7] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i + 1][j] == 1
                    and current_wiring[i + 2][j] == 4
                    and current_wiring[i + 3][j] == 2
                ):
                    vect[8] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i + 1][j] == 4
                    and current_wiring[i + 2][j] == 1
                    and current_wiring[i + 3][j] == 2
                ):
                    vect[9] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 3
                    and current_wiring[i + 2][j] == 4
                    and current_wiring[i + 3][j] == 3
                ):
                    vect[10] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 4
                    and current_wiring[i + 2][j] == 3
                    and current_wiring[i + 3][j] == 3
                ):
                    vect[11] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 1
                    and current_wiring[i + 2][j] == 2
                    and current_wiring[i + 3][j] == 3
                ):
                    vect[12] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 2
                    and current_wiring[i + 2][j] == 1
                    and current_wiring[i + 3][j] == 3
                ):
                    vect[13] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 2
                    and current_wiring[i + 2][j] == 4
                    and current_wiring[i + 3][j] == 3
                ):
                    vect[14] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 4
                    and current_wiring[i + 2][j] == 2
                    and current_wiring[i + 3][j] == 3
                ):
                    vect[15] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 1
                    and current_wiring[i + 2][j] == 4
                    and current_wiring[i + 3][j] == 3
                ):
                    vect[16] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i + 1][j] == 4
                    and current_wiring[i + 2][j] == 1
                    and current_wiring[i + 3][j] == 3
                ):
                    vect[17] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i + 1][j] == 2
                    and current_wiring[i + 2][j] == 3
                    and current_wiring[i + 3][j] == 4
                ):
                    vect[18] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i + 1][j] == 3
                    and current_wiring[i + 2][j] == 2
                    and current_wiring[i + 3][j] == 4
                ):
                    vect[19] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i + 1][j] == 1
                    and current_wiring[i + 2][j] == 2
                    and current_wiring[i + 3][j] == 4
                ):
                    vect[20] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i + 1][j] == 2
                    and current_wiring[i + 2][j] == 1
                    and current_wiring[i + 3][j] == 4
                ):
                    vect[21] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i + 1][j] == 3
                    and current_wiring[i + 2][j] == 1
                    and current_wiring[i + 3][j] == 4
                ):
                    vect[22] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i + 1][j] == 1
                    and current_wiring[i + 2][j] == 3
                    and current_wiring[i + 3][j] == 4
                ):
                    vect[23] += 1
    for l in range(4):
        for i in range(len(current_wiring)):
            for j in range(l, len(current_wiring) - 3, 4):
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i][j + 1] == 2
                    and current_wiring[i][j + 2] == 3
                    and current_wiring[i][j + 3] == 1
                ):
                    vect[0] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i][j + 1] == 3
                    and current_wiring[i][j + 2] == 2
                    and current_wiring[i][j + 3] == 1
                ):
                    vect[1] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i][j + 1] == 2
                    and current_wiring[i][j + 2] == 4
                    and current_wiring[i][j + 3] == 1
                ):
                    vect[2] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i][j + 1] == 4
                    and current_wiring[i][j + 2] == 2
                    and current_wiring[i][j + 3] == 1
                ):
                    vect[3] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i][j + 1] == 3
                    and current_wiring[i][j + 2] == 4
                    and current_wiring[i][j + 3] == 1
                ):
                    vect[4] += 1
                if (
                    current_wiring[i][j] == 1
                    and current_wiring[i][j + 1] == 4
                    and current_wiring[i][j + 2] == 3
                    and current_wiring[i][j + 3] == 1
                ):
                    vect[5] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i][j + 1] == 1
                    and current_wiring[i][j + 2] == 3
                    and current_wiring[i][j + 3] == 2
                ):
                    vect[6] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i][j + 1] == 3
                    and current_wiring[i][j + 2] == 1
                    and current_wiring[i][j + 3] == 2
                ):
                    vect[7] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i][j + 1] == 1
                    and current_wiring[i][j + 2] == 4
                    and current_wiring[i][j + 3] == 2
                ):
                    vect[8] += 1
                if (
                    current_wiring[i][j] == 2
                    and current_wiring[i][j + 1] == 4
                    and current_wiring[i][j + 2] == 1
                    and current_wiring[i][j + 3] == 2
                ):
                    vect[9] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 3
                    and current_wiring[i][j + 2] == 4
                    and current_wiring[i][j + 3] == 3
                ):
                    vect[10] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 4
                    and current_wiring[i][j + 2] == 3
                    and current_wiring[i][j + 3] == 3
                ):
                    vect[11] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 1
                    and current_wiring[i][j + 2] == 2
                    and current_wiring[i][j + 3] == 3
                ):
                    vect[12] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 2
                    and current_wiring[i][j + 2] == 1
                    and current_wiring[i][j + 3] == 3
                ):
                    vect[13] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 2
                    and current_wiring[i][j + 2] == 4
                    and current_wiring[i][j + 3] == 3
                ):
                    vect[14] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 4
                    and current_wiring[i][j + 2] == 2
                    and current_wiring[i][j + 3] == 3
                ):
                    vect[15] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 1
                    and current_wiring[i][j + 2] == 4
                    and current_wiring[i][j + 3] == 3
                ):
                    vect[16] += 1
                if (
                    current_wiring[i][j] == 3
                    and current_wiring[i][j + 1] == 4
                    and current_wiring[i][j + 2] == 1
                    and current_wiring[i][j + 3] == 3
                ):
                    vect[17] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i][j + 1] == 2
                    and current_wiring[i][j + 2] == 3
                    and current_wiring[i][j + 3] == 4
                ):
                    vect[18] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i][j + 1] == 3
                    and current_wiring[i][j + 2] == 2
                    and current_wiring[i][j + 3] == 4
                ):
                    vect[19] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i][j + 1] == 1
                    and current_wiring[i][j + 2] == 2
                    and current_wiring[i][j + 3] == 4
                ):
                    vect[20] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i][j + 1] == 2
                    and current_wiring[i][j + 2] == 1
                    and current_wiring[i][j + 3] == 4
                ):
                    vect[21] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i][j + 1] == 3
                    and current_wiring[i][j + 2] == 1
                    and current_wiring[i][j + 3] == 4
                ):
                    vect[22] += 1
                if (
                    current_wiring[i][j] == 4
                    and current_wiring[i][j + 1] == 1
                    and current_wiring[i][j + 2] == 3
                    and current_wiring[i][j + 3] == 4
                ):
                    vect[23] += 1
    return one_hot


# creating a data set of 1000 inputs and outputs
X = []
Y = []
X_v = []
Y_v = []
# 500 - 1
# 1000 - 1
# 2500 - 1
# 5000 - 1
total_data_size = 5000
training_data_size = (total_data_size * 9) // 10
validation_data_size = total_data_size // 10
testing_data_size = 100
training_epochs = 100
for i in range(training_data_size):
    currentWiring = WiringDiagram(0)
    for j in range(4):
        flattenedFeatureArray = extractFeatures(np.rot90(currentWiring.diagram, j))
        flattenedFeatureArray.insert(0, 1)
        X.append(flattenedFeatureArray)
        Y.append(currentWiring.dangerous)
for i in range(validation_data_size):
    currentWiring = WiringDiagram(0)
    for j in range(4):
        flattenedFeatureArray = extractFeatures(np.rot90(currentWiring.diagram, j))
        flattenedFeatureArray.insert(0, 1)
        X_v.append(flattenedFeatureArray)
        Y_v.append(currentWiring.dangerous)
X = np.array(X)
Y = np.array(Y)
X_v = np.array(X_v)
Y_v = np.array(Y_v)
X_Train = X
Y_Train = Y

model = LogisticalRegression(X_Train, Y_Train, 0.01, training_epochs, X_v, Y_v)

correct_cnt = 0
for i in range(testing_data_size):
    currentWiring = WiringDiagram(0)
    flattenedFeatureArray = extractFeatures(currentWiring.diagram)
    flattenedFeatureArray.insert(0, 1)
    prediction = model.predict(model.weights, flattenedFeatureArray)
    correct_cnt += 1 if prediction == currentWiring.dangerous else 0
print(correct_cnt / testing_data_size)
model.graph_loss_over_time()

from WiringDiagram import WiringDiagram
import numpy as np
from LogisticalRegression import LogisticalRegression


# loops over the wire diagram and creates one hot encoded array of features
def extractFeatures(current_wiring):
    one_hot = []
    for i in range(len(current_wiring)):
        for j in range(len(current_wiring)):
            vect = [0 for _ in range(4)]
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
    return one_hot


# creating a data set of 1000 inputs and outputs
X = []
Y = []
training_data_size = 2000
testing_data_size = 0.20 * training_data_size
training_epochs = 50
for i in range(training_data_size):
    currentWiring = WiringDiagram()
    flattenedFeatureArray = extractFeatures(currentWiring.diagram)
    flattenedFeatureArray.insert(0, 1)
    # currentWiring.print()
    # print(flattenedFeatureArray)
    X.append(flattenedFeatureArray)
    Y.append(currentWiring.dangerous)
X = np.array(X)
# print(X.shape)
Y = np.array(Y)
X_Train = X
Y_Train = Y

model1 = LogisticalRegression(X_Train, Y_Train, 0.01, 150)

correct_cnt = 0
loss = 0
for i in range(int(testing_data_size)):
    currentWiring = WiringDiagram()
    flattenedFeatureArray = extractFeatures(currentWiring.diagram)
    flattenedFeatureArray.insert(0, 1)
    prediction = model1.predict(model1.weights, flattenedFeatureArray)

    loss += model1.loss(model1.weights, flattenedFeatureArray, currentWiring.dangerous)
    correct_cnt += 1 if prediction == currentWiring.dangerous else 0
print(correct_cnt / testing_data_size)
model1.graph_loss_over_time()

import numpy as np
import random


# Wire Guide
# No Wire = 0
# Red = 1
# Blue = 2
# Green = 3
# Yellow = 4
class WiringDiagram:
    def __init__(self, type):
        self.diagram = np.zeros((20, 20))
        # 0 if it is not dangerous and 1 if it is
        self.dangerous = 0
        self.second_last = None
        if type == 0:
            self.generate()
        else:
            self.generate_dangerous()

    def generate(self):
        rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        colors = [1, 2, 3, 4]

        # color the firstRow
        firstRow = random.choice(rows)
        rows.remove(firstRow)
        firstColor = random.choice(colors)
        colors.remove(firstColor)
        if firstColor == 1 and 4 in colors:
            self.dangerous = 1

        for i in range(20):
            self.diagram[firstRow - 1, i] = firstColor

        # color the firstCol
        firstCol = random.choice(cols)
        cols.remove(firstCol)
        secondColor = random.choice(colors)
        colors.remove(secondColor)
        if secondColor == 1 and 4 in colors:
            self.dangerous = 1

        for i in range(20):
            self.diagram[i, firstCol - 1] = secondColor

        # color the secondRow
        secondRow = random.choice(rows)
        rows.remove(secondRow)
        thirdColor = random.choice(colors)
        colors.remove(thirdColor)
        if thirdColor == 1 and 4 in colors:
            self.dangerous = 1

        for i in range(20):
            self.diagram[secondRow - 1, i] = thirdColor

        # color the secondCol
        secondCol = random.choice(cols)
        cols.remove(secondCol)
        fourthColor = random.choice(colors)
        colors.remove(fourthColor)

        for i in range(20):
            self.diagram[i, secondCol - 1] = fourthColor

    def generate_dangerous(self):
        rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        colors = [1, 2, 3]

        # color the firstRow
        firstRow = random.choice(rows)
        rows.remove(firstRow)
        firstColor = random.choice(colors)
        colors.remove(firstColor)
        if firstColor == 1:
            colors.append(4)

        for i in range(20):
            self.diagram[firstRow - 1, i] = firstColor

        # color the firstCol
        firstCol = random.choice(cols)
        cols.remove(firstCol)
        secondColor = random.choice(colors)
        colors.remove(secondColor)
        if secondColor == 1:
            colors.append(4)

        for i in range(20):
            self.diagram[i, firstCol - 1] = secondColor

        # color the secondRow
        secondRow = random.choice(rows)
        rows.remove(secondRow)
        thirdColor = random.choice(colors)
        if thirdColor == 1:
            self.second_last = [1, 0, 0, 0]
        elif thirdColor == 2:
            self.second_last = [0, 1, 0, 0]
        elif thirdColor == 3:
            self.second_last = [0, 0, 1, 0]
        elif thirdColor == 4:
            self.second_last = [0, 0, 0, 1]
        colors.remove(thirdColor)
        if thirdColor == 1:
            colors.append(4)

        for i in range(20):
            self.diagram[secondRow - 1, i] = thirdColor

        # color the secondCol
        secondCol = random.choice(cols)
        cols.remove(secondCol)
        fourthColor = random.choice(colors)
        colors.remove(fourthColor)

        for i in range(20):
            self.diagram[i, secondCol - 1] = fourthColor
        self.dangerous = 1

    def print(self):
        for i in range(20):
            for j in range(20):
                if self.diagram[i][j] == 0:
                    print("⬜", end="", sep="")
                elif self.diagram[i][j] == 1:
                    print("🟥", end="", sep="")
                elif self.diagram[i][j] == 2:
                    print("🟦", end="", sep="")
                elif self.diagram[i][j] == 3:
                    print("🟩", end="", sep="")
                else:
                    print("🟨", end="", sep="")
            print(end="\n")
        print(self.dangerous)
        print(end="\n")

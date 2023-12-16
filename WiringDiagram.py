import numpy as np
import random

#Wire Guide
#No Wire = 0
#Red = 1
#Green = 2
#Blue = 3
#Yellow = 4
class WiringDiagram:
 def __init__(self):
    self.diagram = np.zeros((20, 20))
    #0 if it is not dangerous and 1 if it is
    self.dangerous = 0
    self.generate()

 def generate(self):
    rows = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    colors = [1,2,3,4]
    
    #color the firstRow
    firstRow = random.choice(rows)
    rows.remove(firstRow)
    firstColor = random.choice(colors)
    colors.remove(firstColor)
    if firstColor == 1 and 4 in colors:
        self.dangerous = 1
    
    for i in range(20):
        self.diagram[firstRow-1, i] = firstColor
    
    #color the firstCol
    firstCol = random.choice(cols)
    cols.remove(firstCol)
    secondColor = random.choice(colors)
    colors.remove(secondColor)
    if secondColor == 1 and 4 in colors:
        self.dangerous = 1
    
    for i in range(20):
        self.diagram[i, firstCol-1] = secondColor
    
    #color the secondRow
    secondRow = random.choice(rows)
    rows.remove(secondRow)
    thirdColor = random.choice(colors)
    colors.remove(thirdColor)
    if thirdColor == 1 and 4 in colors:
        self.dangerous = 1
    
    for i in range(20):
        self.diagram[secondRow-1, i] = thirdColor
    
    #color the secondCol
    secondCol = random.choice(cols)
    cols.remove(secondCol)
    fourthColor = random.choice(colors)
    colors.remove(fourthColor)
    
    for i in range(20):
        self.diagram[i, secondCol-1] = fourthColor

    
    
    
    


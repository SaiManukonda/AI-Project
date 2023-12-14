import numpy as np
import random

#Wire Guide
#No Wire = 0
#Red = 1
#Blue = 2
#Yellow = 3
#Green = 4
class WiringDiagram:
 def __init__(self):
    self.diagram = np.zeros((20, 20))
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
    
    for i in range(20):
        self.diagram[firstRow-1, i] = firstColor
    
    #color the firstCol
    firstCol = random.choice(cols)
    cols.remove(firstCol)
    secondColor = random.choice(colors)
    colors.remove(secondColor)
    
    for i in range(20):
        self.diagram[i, firstCol-1] = secondColor
    
    #color the secondRow
    secondRow = random.choice(rows)
    rows.remove(secondRow)
    thirdColor = random.choice(colors)
    colors.remove(thirdColor)
    
    for i in range(20):
        self.diagram[secondRow-1, i] = thirdColor
    
    #color the secondCol
    secondCol = random.choice(cols)
    cols.remove(secondCol)
    fourthColor = random.choice(colors)
    colors.remove(fourthColor)
    
    for i in range(20):
        self.diagram[i, secondCol-1] = fourthColor
    
    print(self.diagram)

test = WiringDiagram()
    
    
    
    


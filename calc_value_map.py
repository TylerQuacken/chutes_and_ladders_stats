import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

nStates = 106
tileMap = np.arange(nStates)
mapFrom =   [1,  4,  9,  21, 28, 36, 48, 49, 51, 56, 62, 64, 71, 80,  87, 93, 95, 98]
mapTo =     [38, 14, 31, 42, 84, 44, 26, 11, 67, 53, 18, 60, 91, 100, 24, 73, 76, 78]
tileMap[mapFrom] = mapTo
tileMap[101:106] = 99

valueMap = np.zeros(nStates) + 100
newVal = np.zeros(nStates)

convergeTol = 0.1
totalChange = 100

while totalChange > convergeTol:
    for i in reversed(range(100)):
        newVal[i] = np.sum(valueMap[tileMap[i+1:i+6]])*1/6 *0.99
                # print("{}: {}".format(i, tileMap[i+1:i+6]))
    
    newVal[100] = 100
    newVal[80] = 100
    newVal[101:] = newVal[99]
    # newVal[101:] = 100

    totalChange = np.sum(abs(newVal - valueMap))
    valueMap = deepcopy(newVal)
    print(totalChange)

plt.figure(1)
plt.plot(valueMap)
plt.show()
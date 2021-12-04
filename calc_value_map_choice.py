import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

nStates = 109
tileMap = np.arange(nStates)
mapFrom = [1,  4,  9,  21, 28, 36, 48, 49, 51, 56, 62, 64, 71, 80,  87, 93, 95, 98]
mapTo =   [38, 14, 31, 42, 84, 44, 26, 11, 67, 53, 18, 60, 91, 100, 24, 73, 76, 78]
tileMap[mapFrom] = mapTo
tileMap[101:106] = 99

winReward = 100
valueMap = np.zeros(nStates) + winReward
newVal = np.zeros(nStates)
policy = np.zeros(nStates)

convergeTol = 0.001
totalChange = 100
actionProb = np.array([0.25, 0.5, 0.25])
maxIter = 1000
iter = 0

while totalChange > convergeTol and iter < maxIter:
    iter += 1
    for i in reversed(range(100)):
        choices = np.zeros(6)
        for j in range(0, 6):
            choices[j] = np.sum(valueMap[tileMap[i+j:i+j+3]] * actionProb)

        bestChoice = np.max(choices)
        policy[i] = np.argmax(choices)
        newVal[i] = bestChoice * 0.9
                # print("{}: {}".format(i, tileMap[i+1:i+6]))
    
    newVal[100] = winReward
    newVal[80] = winReward
    newVal[101:] = newVal[99]
    # newVal[101:] = 100

    totalChange = np.sum(abs(newVal - valueMap))
    valueMap = deepcopy(newVal)
    print(totalChange)

fig, axs = plt.subplots(2)

axs[0].plot(valueMap)
axs[0].set_ylabel('Value')
axs[1].set_ylabel('Optimal Action')
axs[1].plot(policy + 1)
plt.show()
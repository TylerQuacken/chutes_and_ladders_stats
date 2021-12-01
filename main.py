import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

numParallel = 1000
numRolls = 10000
tileMap = np.arange(101)
mapFrom =   [1,  4,  9,  21, 28, 36, 48, 49, 51, 56, 62, 64, 71, 80, 87, 93, 95, 98, 100]
mapTo =     [38, 14, 31, 42, 84, 44, 26, 11, 67, 53, 18, 60, 91, 0,  24, 73, 76, 78, 0]
tileMap[mapFrom] = mapTo
wins = 0
winLength = np.array([])
rollouts = np.array([], dtype=np.ndarray)
currentPlayLength = np.zeros(numParallel, 'int')

heatMap = np.zeros(101)

position = np.zeros(numParallel, 'int')

for i in tqdm(range(numRolls)):
    # roll for each sim
    roll = np.random.randint(1, 6, numParallel)

    # Move each piece
    position += roll

    # Enforce that a win means a move to tile 100 exactly
    tooFar = position > 100
    position[tooFar] -= roll[tooFar]

    # update heatMap
    for pos in position:
        heatMap[pos] += 1

    # perform chutes + ladders
    position = tileMap[position]
    
    # check for wins
    currentPlayLength += 1
    wins += np.sum(position == 0)
    winLength = np.append(winLength, currentPlayLength[position == 0])
    currentPlayLength[position == 0] = 0



heatMap /= numRolls * numParallel

print(numRolls * numParallel / wins)

plt.figure(1)
plt.plot(heatMap)

plt.figure(2)
plt.hist(winLength, bins = 100)
# plt.boxplot(winLength)
plt.show()
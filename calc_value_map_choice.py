import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

nStates = 107   # the number of possible states attainable immediately after a roll
tileMap = np.arange(nStates)    # initialize the tilemap
mapFrom = [1,  4,  9,  21, 28, 36, 48, 49, 51, 56, 62, 64, 71, 80,  87, 93, 95, 98]
mapTo =   [38, 14, 31, 42, 84, 44, 26, 11, 67, 53, 18, 60, 91, 100, 24, 73, 76, 78]
tileMap[mapFrom] = mapTo    # mapFrom is the start of a chute/ladder, mapTo is its destination
tileMap[101:] = 99   # overshooting tile 100 leaves the player on tile 99

winReward = 100 # the reward for winning - this is arbitrary
valueMap = np.zeros(nStates)    # the value of each tile, 1:nStates
newVal = np.zeros(nStates)      # for calculating the change in value in a given timestep
policy = np.zeros(nStates)      # the best action to take for each tile

convergeTol = 0.001     # when the total change in value is this low, we stop
totalChange = 100       # initialize to an arbitrary high value
actionProb = np.array([0.25, 0.5, 0.25])    # the probability of [undershoot, perfect, overshoot]
discountFactor = 0.9    # this controls how quickly the "win reward" decays in the value map
maxIter = 1000      # maximum number of value iterations
iter = 0    # initialize iteration counter

# loop until we converge or time out
while totalChange > convergeTol and iter < maxIter:
    iter += 1
    for i in reversed(range(100)):      # step backwards through tiles
        choices = np.zeros(6)           # keep track of the value of each action
        for j in range(0, 6):           # iterate through possible actions to find the best action from a tile
            # the value achieved with each action is a probability-weighted sum of the value of each tile
            # that can be reached by taking a given action
            choices[j] = np.sum(valueMap[tileMap[i+j : i+j+3]] * actionProb)

        bestChoice = np.max(choices)    # the value of the best action
        policy[i] = np.argmax(choices)  # the action resulting in the best value
        newVal[i] = bestChoice * discountFactor     # set the new value of tile i
    
    newVal[100] = winReward     # ensure the win reward is unchanged
    newVal[80] = winReward      # tile 80 has a ladder to 100, so it's basically a win
    newVal[101:] = newVal[99]   # tiles 101 to 106 just map back to tile 99

    totalChange = np.sum(abs(newVal - valueMap))    # figure out how much the value map changed
    valueMap = deepcopy(newVal)         # set the new value map
    print(totalChange)

# plot everything, excluding the slush tiles (101 - 106)
fig, axs = plt.subplots(2)

axs[0].plot(valueMap[:101])
axs[0].set_ylabel('Value')
axs[1].set_ylabel('Optimal Action')
axs[1].set_xlabel('Tile')
axs[1].plot(policy[:101] + 1)   # add 1 because action '0' moves you forward 1 tile
fig.suptitle('Optimal Policy for Chutes and Ladders')
plt.show()
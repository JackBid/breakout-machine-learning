import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from agents.supervised import SupervisedAgent

#agent = SupervisedAgent('test', False, True)
agent = SupervisedAgent('fc364', True, True)
agent.train(5)

# Get which action is selected based on inputs
def getAction(ballMid, ballY, paddleMid):
    tensorInput = torch.tensor([ballMid, ballY, paddleMid])
    outputs = agent.net(tensorInput.float())

    action = int(torch.max(outputs, 0)[1])
    
    return action

# Get how confidence the AI is for each move
# normalize the tensor output to see how confident it is
def getActionConfidence(ballMid, ballY, paddleMid):
    tensorInput = torch.tensor([ballMid, ballY, paddleMid])
    outputs = agent.net(tensorInput.float())
    softmax = F.softmax(outputs)
    #print(softmax)
    maxVal = float(torch.max(softmax, 0)[0])
   # print(maxVal)
    #print()
    return maxVal

ballY = 128

actionGrid = np.zeros((160, 160))
confidenceGrid = np.zeros((160, 160))

for ballMid in range(0, 160):
    for paddleMid in range(0, 160):
        actionGrid[ballMid][paddleMid] = getAction(ballMid, ballY, paddleMid) / 4
        confidenceGrid[ballMid][paddleMid] = getActionConfidence(ballMid, ballY, paddleMid)

fig, ax = plt.subplots()
plt.ylabel('x coordinate of ball mid-point')
plt.xlabel('x coordinate of paddle mid-point')
ax.set_ylim(0, 160)
ax.imshow(confidenceGrid, interpolation='nearest')
plt.show()





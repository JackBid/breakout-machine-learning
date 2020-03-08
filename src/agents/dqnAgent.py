import numpy as np
import gym
import torch
import math
import time
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.util import Util
from nets import FullyConnected, Test

class DQNAgent():

    def __init__(self):
        super().__init__()

        self.util = Util()

        self.net = FullyConnected(128, 10)
        self.net.float()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.net.cuda()
            #self.net.load_state_dict(torch.load('../res/models/transfer.pth'))
        else:
            self.device = torch.device('cpu')
            #self.net.load_state_dict(torch.load('../res/models/transfer.pth', map_location=('cpu')))

        print('device: ' + str(self.device) + '\n')    

        # Create a gym environment (game environment)
        self.env = gym.make('Breakout-ram-v0')
        self.env.frameskip = 0
        
        self.explorationRate = 0.05

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())

        self.replayMemory = []
        self.explorationRate = 1.0
        self.decayRate = 0.001
        

    def getReplayMemory(self, numGames):

        for i in range(0, numGames):

            observation = self.env.reset()
            t = 0

            while True:

                #self.env.render()

                ram = self.util.observationToTensor(observation)

                ballY = int(observation[101])

                if random.uniform(0,1) < self.explorationRate:
                    action = random.randrange(4)
                elif t == 0 or ballY > 200 or ballY == 0:
                    action = 1
                else:
                    action = self.util.tensorAction(self.net, ram)

                newObservation, reward, done, info = self.env.step(action)

                self.replayMemory.append((self.util.observationToTensor(observation), action, reward, self.util.observationToTensor(newObservation), done))

                observation = newObservation

                t += 1
                self.explorationRate -= self.explorationRate * self.decayRate

                if done:
                    break

    def train(self):

        self.getReplayMemory(5)

        for replay in self.replayMemory:
            
            observation = replay[0]
            action = replay[1]
            reward = replay[2]
            nextObservation = replay[3]
            done = replay[4]

            output = self.util.getOutput(self.net, observation)
            q_value = torch.max(output)

            target = reward + 0.5 * self.util.tensorAction(self.net, nextObservation)
            
            self.optimizer.zero_grad()

            loss = self.criterion(q_value, torch.tensor(target))
            loss.backward()
            self.optimizer.step()
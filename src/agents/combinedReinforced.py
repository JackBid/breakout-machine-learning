import numpy as np
import gym
import torch
import math
import time
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.supervised import SupervisedAgent
from nets import FullyConnected, Test

class CombinedReinforcedAgent():

    def __init__(self, save=False):
        super().__init__()

        self.net = FullyConnected(128, 10)
        self.net.float()

        self.save = save

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.net.cuda()
            self.net.load_state_dict(torch.load('../res/models/combined.pth'))
        else:
            self.device = torch.device('cpu')
            self.net.load_state_dict(torch.load('../res/models/combined.pth', map_location=('cpu')))

        print('device: ' + str(self.device) + '\n')    

        # Create a gym environment (game environment)
        self.env = gym.make('Breakout-ram-v0')
        self.env.frameskip = 0
        
        self.trainingData = self.loadTrainingData('../res/training data/ram.txt')
        self.testingData = self.loadTestingData('../res/training data/action.txt')

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

        # Create supervised agent
        self.supervisedAgent = SupervisedAgent()

    def loadTrainingData(self, path):
        training = []

        trainingFile = open(path, "r")

        for ram in trainingFile:
            data = [int(numeric_string) for numeric_string in ram.split(' ')[:-1]]
            training.append(torch.tensor(data, dtype=torch.uint8))
        
        return training

    def loadTestingData(self, path):
        testing = []

        testingFile = open(path, "r")

        for num in testingFile.readline().split(' ')[:-1]:
            data = int(num)
            testing.append(torch.torch.LongTensor([data]))
        
        return testing
    
    def getSupervisedAgentAction(self, observation):
        action = self.supervisedAgent.action(observation)
        return action
    
    def getOutput(self, tensorIn):
        output = self.net(tensorIn)
        return output

    def action(self, tensorIn):
        output = self.net(tensorIn)

        maxVal = torch.max(output, 0)
        return int(maxVal[1])

    # Based on where the ball landed in relation to the paddle, find the target
    def getTarget(self, paddleMid, ballMid):
        arr = []

        # If the ball hits the paddle, remain in the same place
        if abs(paddleMid - ballMid) < 7:
            arr = [1.0, 0.0, 0.0, 0.0]
        # paddle left of ball so move right
        if paddleMid < ballMid:
            arr = [0.0, 0.0, 1.0, 0.0]
        # paddle right of ball so move left
        elif paddleMid > ballMid:
            arr = [0.0, 0.0, 0.0, 1.0]
            
        return torch.tensor(arr, device=self.device)
    
    def supervisedLearn(self, iterations):
        for iteration in range(0, iterations):

            running_loss = 0.0
            for i in range(0, len(self.trainingData)):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Get the outputs and target
                outputs = self.net(self.trainingData[i].float())
                target = self.testingData[i].clone()

                outputs = outputs.unsqueeze(dim=0)
                #print(outputs)

                loss = self.criterion(outputs, target)
                #print('loss: ' + str(loss))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                    (iteration + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                    if self.save:
                        torch.save(self.net.state_dict(), '../res/models/combined.pth')

    def selfLearn(self, iterations):

        maxReward = 0
        total = 0
        startTime = time.time()

        # How many iterations of the game should be played
        for i_episode in range(iterations):

            observation = self.env.reset()

            iteration_reward = 0
            t = 0

            # One game iteration
            while True:

                self.env.render()

                paddleMid = int(observation[72]) + 8
                ballMid = int(observation[99]) + 1
                ballY = int(observation[101])
                
                # Create tensor for observation
                supervisedAgentAction = float(self.getSupervisedAgentAction(observation))
                tensorIn = torch.from_numpy(np.append(observation, supervisedAgentAction)).float().to(self.device)
                
                if t == 0 or ballY > 200 or ballY <= 0:
                    action = 1#config.ACTION_FIRE
                else:
                    action = self.action(tensorIn)

                observation, reward, done, info = self.env.step(action)

                if ballY <= 25: 
        
                    self.optimizer.zero_grad()
                    
                    output = self.getOutput(tensorIn)
                    target = self.getTarget(paddleMid, ballMid)

                    criterion = nn.MSELoss()
                    loss = criterion(output, target)

                    loss.backward()
                    self.optimizer.step()

                iteration_reward += reward
                t += 1

                if done:
                    break

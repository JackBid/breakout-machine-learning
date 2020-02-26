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

class BasicReinforcedAgent():

    def __init__(self):
        super().__init__()

        self.net = FullyConnected(128, 10)
        self.net.float()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.net.cuda()
            self.net.load_state_dict(torch.load('../res/models/test.pth'))
        else:
            self.device = torch.device('cpu')
            self.net.load_state_dict(torch.load('../res/models/test.pth', map_location=('cpu')))

        print('device: ' + str(self.device) + '\n')    

        # Create a gym environment (game environment)
        self.env = gym.make('Breakout-ram-v0')
        self.env.frameskip = 0
        
        self.explorationRate = 0.05

        self.optimizer = optim.Adam(self.net.parameters())

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

    # Scale RAM values (tensor) to be between 0-1
    def scaleRAM(self, RAM):

        scaled_RAM = []

        for value in RAM:
            scaled_RAM.append(value.item() / 255)

        return torch.tensor(scaled_RAM, requires_grad=True).to(self.device)

    # Get the action the network takes
    def tensorAction(self, tensorIn):
        output = self.net(tensorIn)

        maxVal = torch.max(output, 0)
        return int(maxVal[1])

    # Take observation input and convert to scaled tensor
    def observationToTensor(self, observation):
        observation = self.scaleRAM(observation)
        return torch.tensor(observation, requires_grad=True, device=self.device)

    # Get the output from the network from a particular observation
    def getOutput(self, tensorIn):
        output = self.net(tensorIn)
        return output

    # Get the output from the network from a particular observation scaled
    def getScaledOutput(self, tensorIn):
        output = self.net(tensorIn)
        output = F.softmax(output)
        return output
    
    # Get the action the network takes based on an observation
    def observationAction(self, observation):
        observationTensor = torch.tensor(observation, device=self.device)

        paddleMid = int(observation[72]) + 10
        ballMid = int(observation[99]) + 1
        ballY = int(observation[101])

        tensorInput = torch.tensor([ballMid, paddleMid], device=self.device)
        outputs = self.net(tensorInput.float())

        maxVal = torch.max(outputs, 0)
        return int(maxVal[1])

    def calculateTargetLoss(self, ballMid, paddleMid):
    
        distance = abs(ballMid - paddleMid)
        if distance < 7:
            loss = 0
        else: 
            loss = distance / 160
        return torch.tensor(loss)

    def applyLoss(self, outputs, targetLoss):
        loss = torch.mul(outputs, targetLoss)
        loss = torch.sum(torch.abs(loss))
        return loss
    
    def train(self, iterations):

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
                
                #self.env.render()

                paddleMid = int(observation[72]) + 8
                ballMid = int(observation[99]) + 1
                ballY = int(observation[101])

                ram = self.observationToTensor(observation)
                output = self.getScaledOutput(ram)

                if t == 0 or ballY > 200 or ballY <= 0:
                    action = 1#config.ACTION_FIRE
                else:
                    action = self.tensorAction(ram)

                observation, reward, done, info = self.env.step(action)
                
                if ballY >= 175:
                    #Calculate and apply loss
                    self.optimizer.zero_grad()

                    targetLoss = self.calculateTargetLoss(paddleMid, ballMid)
                    loss = self.applyLoss(output, targetLoss)
                    loss.backward()
                    
                    # print(loss.grad)
                    self.optimizer.step()
                



                iteration_reward += reward
                t += 1

                if done:    
                    break
            
            total += iteration_reward
            if iteration_reward > maxReward:
                maxReward = iteration_reward

            #if i_episode != 0 and i_episode % 1000 == 0:
                
        # Calcuate how long this training took
        elapsed_time = time.time() - startTime
        minutes = math.floor(elapsed_time/60)
        seconds = math.floor(elapsed_time - minutes * 60)

        average_reward = total / iterations
        
        print('average reward: ' + str(average_reward))
        print('max reward: ' + str(maxReward))
        print('Training took: ' + str(minutes) + ':' + str(seconds))
        print()

        #file = open('../res/learning_progress2.txt', 'a+')
        #file.write(str(average_reward) + ' ' + str(maxReward) + '\n')
        
        torch.save(self.net.state_dict(), '../res/models/test.pth')

        maxReward = 0
        total = 0

        startTime = time.time()

                




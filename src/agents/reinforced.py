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
from agents.util import Util
from nets import FullyConnected, Test

class BasicReinforcedAgent():

    def __init__(self):
        super().__init__()

        self.util = Util()

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

    # One game played by the agent
    def gameCycle(self, observation, t, iteration_reward):
        while True:
            self.env.render()

            paddleMid = int(observation[72]) + 8
            ballMid = int(observation[99]) + 1
            ballY = int(observation[101])

            ram = self.util.observationToTensor(observation)
            output = self.util.getScaledOutput(self.net, ram)

            if t == 0 or ballY > 200 or ballY <= 0:
                action = 1#config.ACTION_FIRE
            else:
                action = self.util.tensorAction(self.net, ram)

            observation, reward, done, info = self.env.step(action)
            
            if ballY >= 175:
                self.processLoss(paddleMid, ballMid, output)

            iteration_reward += reward
            t += 1

            if done:    
                return iteration_reward, t


    # Calculate and apply loss
    def processLoss(self, paddleMid, ballMid, output):
        self.optimizer.zero_grad()

        targetLoss = self.util.calculateTargetLoss(paddleMid, ballMid)
        loss = self.util.applyLoss(output, targetLoss)
        loss.backward()
        
        self.optimizer.step()

    # At the end of training (or after fixed number of cycles) print statistics and save model
    def processTraining(self, totalReward, iterations, maxReward, elapsed_time):

        minutes = math.floor(elapsed_time/60)
        seconds = math.floor(elapsed_time - minutes * 60)

        average_reward = totalReward / iterations
                
        print('average reward: ' + str(average_reward))
        print('max reward: ' + str(maxReward))
        print('Training took: ' + str(minutes) + ':' + str(seconds))
        print()

        #file = open('../res/learning_progress2.txt', 'a+')
        #file.write(str(average_reward) + ' ' + str(maxReward) + '\n')
        
        torch.save(self.net.state_dict(), '../res/models/test.pth')

        maxReward = 0
        total = 0

    # Train the agent
    def train(self, iterations):

        maxReward = 0
        totalReward = 0
        startTime = time.time()

        # How many iterations of the game should be played
        for i_episode in range(iterations):

            observation = self.env.reset()

            iteration_reward = 0
            t = 0

            # One complete game
            iteration_reward, t = self.gameCycle(observation, t, iteration_reward)
                
            totalReward += iteration_reward
            if iteration_reward > maxReward:
                maxReward = iteration_reward

            if i_episode != 0 and i_episode % 1000 == 0: 
                # Calcuate how long this training took
                elapsed_time = time.time() - startTime
                self.processTraining(totalReward, iterations, maxReward, elapsed_time)
                startTime = time.time()
        
        # One all games finished process training again
        elapsed_time = time.time() - startTime
        self.processTraining(totalReward, iterations, maxReward, elapsed_time)
        startTime = time.time()

                




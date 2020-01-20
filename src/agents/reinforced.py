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

from nets import FC364, Test

class BasicReinforcedAgent():

    def __init__(self):
        super().__init__()
        
        self.cuda = torch.device('cuda')

        self.net = FC364()
        self.net.float()
        self.net.cuda()

        self.net.load_state_dict(torch.load('../res/models/game_based_adjusted.pth'))

        # Create a gym environment (game environment)
        self.env = gym.make('Breakout-ram-v0')
        self.env.frameskip = 0
        
        self.explorationRate = 0.05

        self.optimizer = optim.Adam(self.net.parameters())

    # Based on where the ball landed in relation to the paddle, find the target
    def getTarget(self, paddleMid, ballMid):
        arr = []

        # If the ball hits the paddle, remain in the same place
        if abs(paddleMid - ballMid) < 8:
            arr = [1.0, 0.0, 0.0, 0.0]
        # paddle left of ball so move right
        if paddleMid < ballMid:
            arr = [0.0, 0.0, 1.0, 0.0]
        # paddle right of ball so move left
        elif paddleMid > ballMid:
            arr = [0.0, 0.0, 0.0, 1.0]
            
        return torch.tensor(arr, device=self.cuda)

    # Get the output from the network from a particular observation
    def getOutput(self, observation):
        observationTensor = torch.tensor(observation, device=self.cuda)

        paddleMid = int(observation[72]) + 8
        ballMid = int(observation[99]) + 1
        ballY = int(observation[101])

        tensorInput = torch.tensor([ballMid, ballY, paddleMid], device=self.cuda)
        outputs = self.net(tensorInput.float())

        return outputs
    
    # Get the action the network takes based on an observation
    def action(self, observation):
        observationTensor = torch.tensor(observation, device=self.cuda)

        paddleMid = int(observation[72]) + 8
        ballMid = int(observation[99]) + 1
        ballY = int(observation[101])

        tensorInput = torch.tensor([ballMid, ballY, paddleMid], device=self.cuda)
        outputs = self.net(tensorInput.float())

        maxVal = torch.max(outputs, 0)
        return int(maxVal[1])
    
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
                
                # If the ball is off screen or the game is at the start, fire
                # Otherwise use the agent to decide action
                if random.random() < self.explorationRate:
                    action = random.randint(0, 3)
                elif t == 0 or ballY > 200 or ballY <= 0:
                    action = 1#config.ACTION_FIRE
                else:
                    action = self.action(observation)

                observation, reward, done, info = self.env.step(action)

                if ballY <= 25: 
        
                    self.optimizer.zero_grad()
                    
                    output = self.getOutput(observation)
                    target = self.getTarget(paddleMid, ballMid)

                    criterion = nn.MSELoss()
                    loss = criterion(output, target)

                    loss.backward()
                    self.optimizer.step()

                iteration_reward += reward
                t += 1

                if done:
                    break
            
            total += iteration_reward
            if iteration_reward > maxReward:
                maxReward = iteration_reward

            if i_episode != 0 and i_episode % 1000 == 0:
                
                # Calcuate how long this training took
                elapsed_time = time.time() - startTime
                minutes = math.floor(elapsed_time/60)
                seconds = math.floor(elapsed_time - minutes * 60)

                average_reward = total / 1000
                
                print('average reward: ' + str(average_reward))
                print('max reward: ' + str(maxReward))
                print('Training took: ' + str(minutes) + ':' + str(seconds))
                print()

                file = open('../res/learning_progress.txt', 'a+')
                file.write(str(average_reward) + ' ' + str(maxReward) + '\n')
                
                torch.save(self.net.state_dict(), '../res/models/game_based_adjusted.pth')

                maxReward = 0
                total = 0

                startTime = time.time()



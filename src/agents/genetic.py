import numpy as np
import torch
import gym
import math
import config
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nets import FC364, Test

class ReinforcedAgent():

    def __init__(self):
        super().__init__()

        self.net = FC364()
        self.net = self.net.float()

        # Create a gym environment (game environment)
        self.env = gym.make('Breakout-ram-v0')
        self.env.frameskip = 0
    
    def action(self, observation):
        observationTensor = torch.tensor(observation)

        paddleMid = int(observation[72]) + 13
        ballMid = int(observation[99]) + 1
        ballY = int(observation[101])

        tensorInput = torch.tensor([ballMid, ballY, paddleMid])
        outputs = self.net(tensorInput.float())

        maxVal = torch.max(outputs, 0)
        return int(maxVal[1])

    def evaluate(self, fc1_weight, fc2_weight, gamma=1.0):
        fc1_weight = fc1_weight.type(torch.FloatTensor)
        fc2_weight = fc2_weight.type(torch.FloatTensor)

        #Set weights
        self.net.fc1.weight = torch.nn.Parameter(fc1_weight)
        self.net.fc2.weight = torch.nn.Parameter(fc2_weight)

        episode_return = 0.0
        observation = self.env.reset()
        t = 0

        while True:
            
            ballY = int(observation[101])
                
            # If the ball is off screen or the game is at the start, fire
            # Otherwise use the agent to decide action
            if t == 0 or ballY > 200 or ballY == 0:
                action = config.ACTION_FIRE
            else:
                action = self.action(observation)

            observation, reward, done, _ = self.env.step(action)
            episode_return += reward #* math.pow(gamma, t)

            if done:
                break

            t += 1
        return episode_return
    
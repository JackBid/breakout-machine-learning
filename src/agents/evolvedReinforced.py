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

class EvolvedReinforcedAgent():

    def __init__(self):
        super().__init__()

        self.util = Util()

        self.net = FullyConnected(128, 10)
        self.net.float()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.net.cuda()
            self.net.load_state_dict(torch.load('../res/models/evolved1.pth'))
        else:
            self.device = torch.device('cpu')
            self.net.load_state_dict(torch.load('../res/models/evolved1.pth', map_location=('cpu')))

        print('device: ' + str(self.device) + '\n')    

        # Create a gym environment (game environment)
        self.env = gym.make('Breakout-ram-v0')
        self.env.frameskip = 0
        
        self.explorationRate = 0.05

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())

    # Add random noise to parameters
    def addParamNoise(self, sigma):
        for param in self.net.parameters():
            noise = torch.FloatTensor(param.shape).uniform_(-sigma, sigma).to(self.device)
            param.data = torch.add(param, noise).to(self.device)

    def calculateActionLoss(self, action, target):
        if action == target:
            loss = 0
        elif (action == 2 and target == 3) or (action == 3 and target == 2):
            loss = 0.1
        else:
            loss = 0.05

        return torch.tensor(loss)

    def learnFromReplay(self, newObservations, newActions, newStates):

        for i in range(0, len(newStates)):

            # restore the previous state
            self.env.ale.restoreState(newStates[i])
            observation = newObservations[i]

            # get action
            ram = self.util.observationToTensor(observation)
            ballY = int(observation[101])

           # if ballY > 200 or ballY <= 0:
            #    action = 1
            #else:
            action = self.util.tensorAction(self.net, ram)

            #if action == 1 or newActions[i] == 1:
            #    continue
            self.optimizer.zero_grad()

            output = self.util.getScaledOutput(self.net, ram)
            target = self.util.actionToTensor(newActions[i]).float().to(self.device)
            loss = self.criterion(output, target).to(self.device)
           # if action == 1 and newActions[i] == 1:
            #print(output)
            #print(target)
            #print('action: ' + str(action) + ' target: ' + str(newActions[i]) + ' loss: ' + str(loss))

            loss.backward()
            self.optimizer.step()

    def getParams(self):
        params = []

        for param in self.net.parameters():
            params.append(param.data)
        
        return params

    def restoreParams(self, params):

        i = 0

        for param in self.net.parameters():
            param.data = params[i]
            i += 1

    def replaySample(self, startingState, observations, actions, lives, sampleLength, originalScore):
        
        prevParams = self.getParams()

        self.addParamNoise(0.005)

        # restore the starting state (since death)
        self.env.ale.restoreState(startingState)

        newObservations = []
        newActions = []
        states = []

        didDie = False
        t = 0
        newScore = 0

        observation = observations[0]
        
        while True:
            
            #self.env.render()

            ballY = int(observation[101])
            ram = self.util.observationToTensor(observation)

            states.append(self.env.ale.cloneState())
            newObservations.append(observation)

            if ballY > 200 or ballY <= 0:
                action = 1#config.ACTION_FIRE
                self.restoreParams(prevParams)
                return
            else:
                action = self.util.tensorAction(self.net, ram)

            observation, reward, done, info = self.env.step(action)

            newActions.append(action)

            newScore += reward

            if info['ale.lives'] < lives:
                didDie = True

            t+=1

            if t==sampleLength:
                if not didDie:
                    self.learnFromReplay(newObservations[0:sampleLength], newActions[0:sampleLength], states[0:sampleLength])
                self.restoreParams(prevParams)
                break

            #time.sleep(0.5)

        return

    # One game played by the agent
    def gameCycle(self, iteration):

        iteration_reward = 0
        t = 0
        done = True
        
        record = False
        timeLimit = 0
        sampleScore = 0
        lives = 5
        maxSampleLength = 10
       
        observations = []
        actions = []
        states = []

        observation = self.env.reset()

        while True:

            #self.env.render()

            paddleMid = int(observation[72]) + 8
            ballMid = int(observation[99]) + 1
            ballY = int(observation[101])

            states.append(self.env.ale.cloneState())
            observations.append(observation)

            ram = self.util.observationToTensor(observation)
            output = self.util.getScaledOutput(self.net, ram)

            if t == 0 or ballY > 200 or ballY <= 0:
                action = 1#config.ACTION_FIRE
            else:
                action = self.util.tensorAction(self.net, ram)

            actions.append(action)

            observation, reward, done, info = self.env.step(action)

            if info['ale.lives'] < lives:
                
                if t < maxSampleLength:
                    sampleLength = t
                else:
                    sampleLength = maxSampleLength

                self.replaySample(states[-sampleLength], observations[-sampleLength:], actions[-sampleLength:], lives, sampleLength, sampleScore)
                self.env.ale.restoreState(states[-1])
                observation = observations[-1]
                lives -= 1 
                t = 0
                states = []
                observations = []
                actionSamples = []

            iteration_reward += reward
            sampleScore += reward
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

        #file = open('../res/evolvedProgress.txt', 'a+')
        #file.write(str(average_reward) + ' ' + str(maxReward) + '\n')
        
        torch.save(self.net.state_dict(), '../res/models/evolved1.pth')

        maxReward = 0
        total = 0

    # Train the agent
    def train(self, iterations):

        maxReward = 0
        totalReward = 0
        startTime = time.time()

        # How many iterations of the game should be played
        for i_episode in range(iterations):

            # One complete game
            #print('Starting game: ' + str(i_episode))
            iteration_reward, t = self.gameCycle(i_episode)
            #print('Game ' + str(i_episode) + ' reward: ' + str(iteration_reward) + '\n')
                
            totalReward += iteration_reward
            if iteration_reward > maxReward:
                maxReward = iteration_reward

            if i_episode != 0 and i_episode % 100 == 0: 
                # Calcuate how long this training took
                elapsed_time = time.time() - startTime
                self.processTraining(totalReward, iterations, maxReward, elapsed_time)
                startTime = time.time()
        
        # One all games finished process training again
        elapsed_time = time.time() - startTime
        self.processTraining(totalReward, iterations, maxReward, elapsed_time)
        startTime = time.time()

    def observationAction(self, observation):

        ram = self.util.observationToTensor(observation)
        output = self.util.getScaledOutput(self.net, ram)

        maxVal = torch.max(output, 0)
        return int(maxVal[1])

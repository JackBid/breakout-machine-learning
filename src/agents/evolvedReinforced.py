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
    """ Class for evolved reinforcement agent

    Represents an agent that can train and make actions.

    Has methods for replaying samples of the games,
    calculating loss of moves and taking actions.

    """
    def __init__(self, debug = False, render = False):
        super().__init__()
        
        # Load a util object used for common methods
        self.util = Util()

        # Create a new network
        self.net = FullyConnected(128, 10)
        self.net.float()

        # Where to save and load model weights
        self.saveFile = '../res/models/reinforced.pth'
        
        # Load weights for the network using cuda or cpu
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.net.cuda()
            self.net.load_state_dict(torch.load(self.saveFIle))
        else:
            self.device = torch.device('cpu')
            self.net.load_state_dict(torch.load(self.saveFile, map_location=('cpu')))

        # Create a gym environment (game environment)
        self.env = gym.make('Breakout-ram-v0')
        self.replayEnv = gym.make('Breakout-ram-v0')
        self.env.frameskip = 0
        self.replayEnv.frameskip = 0

        # Learning parameters
        self.sampleRate = 0.005
        self.maxSampleLength = 30
        self.networkParameterNoise = 0.015
        self.minimumNetworkParameterNoise = 0.0015
        self.decayRate = 0.999
        self.deathPenalty = 100
        self.learnReplayRate = 3
        self.frameBuffer = 10
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())

        # Boolean for debugging statements
        self.debug = debug
        self.render = render
        self.writeProgress = False


    def addParamNoise(self, sigma):
        """Add noise to the network parameters
        This is used for exploring new random actions.
        """
        for param in self.net.parameters():
            noise = torch.FloatTensor(param.shape).uniform_(-sigma, sigma).to(self.device)
            param.data = torch.add(param, noise).to(self.device)


    def learnFromReplay(self, newObservations, newActions, newStates):
        """ Learn from a successful replay
        Given list of observations, actions and states, alter the current model.

        newObservations : list = List of observations where each observation is a list of ints in range[0, 255]
        newActions : list = List of actions where each action is an integer from {0,1,2,3}
        newStates : list = List of ints where each int is represents an OpenAI state than can be loaded 

        Learn from replay variables by restoring the state, using the observation as input to the network,
        and comparing the output with newActions.
        """

        if self.debug:
            print('Learning from replay samples...')

        for i in range(0, self.learnReplayRate):
            for i in range(0, len(newStates)):

                # restore the previous state
                self.env.ale.restoreState(newStates[i])
                observation = newObservations[i]

                # get action
                ram = self.util.observationToTensor(observation)
                action = self.util.tensorAction(self.net, ram)

                self.optimizer.zero_grad()

                output = self.util.getScaledOutput(self.net, ram)
                target = self.util.actionToTensor(newActions[i]).float().to(self.device)
                loss = self.criterion(output, target).to(self.device)

                if self.debug:
                    print('action: ' + str(action) + ' target: ' + str(newActions[i]) + ' loss: ' + str(loss))

                loss.backward()
                self.optimizer.step()


    def getParams(self):
        """ Get the current parameters of the network

        Get the current parameters of the network and return them,
        so they can be saved and reloaded.
        """
        params = []

        for param in self.net.parameters():
            params.append(param.data)
        
        return params


    def restoreParams(self, params):
        """ Restore network parameters

        params : list = List of network parameters to be restored
        """
        i = 0

        for param in self.net.parameters():
            param.data = params[i]
            i += 1


    def replaySample(self, startingState, startingObservation, lives, sampleLength, originalScore):
        """ Replay a sample of the game but randomly explore new moves

        startingState : int = integer representing the state of the game to reload
        startingObservation : list = observation to input to network at start of replay
        lives : int = number of lives left
        sampleLength : int = number of iterations to replay sample for
        originalScore : int = score originally achieved by this agent (starting from the same state for the same sampleLength)
    
        If a greater score is achieved when replaying with exploration rate, then learn from this sample.
        """

        # Save the current network parameters and add noise to the current network
        prevParams = self.getParams()
        self.addParamNoise(self.networkParameterNoise)

        # Decay the parameter noise
        if self.networkParameterNoise > self.minimumNetworkParameterNoise:
            self.networkParameterNoise *= self.decayRate

        # restore the starting state (since death)
        self.replayEnv.reset()
        self.replayEnv.ale.restoreState(startingState)

        # store new observations, actions, states, timesteps and score
        newObservations = []
        newActions = []
        states = []
        t = 0
        newScore = 0

        observation = startingObservation
        
        while True:
            
            if self.render:
                self.replayEnv.render()

            ballY = int(observation[101])
            ram = self.util.observationToTensor(observation)

            states.append(self.env.ale.cloneState())
            newObservations.append(observation)

            action = self.util.tensorAction(self.net, ram)

            observation, reward, done, info = self.replayEnv.step(action)
            newActions.append(action)

            if t <= sampleLength:
                newScore += reward

            # Add penalty for dying
            if info['ale.lives'] < lives:
                newScore -= self.deathPenalty
                lives -= 1

            t += 1
            
            # If the sample is over (after buffer timesteps)
            # If a higher score was achieved learn from replay
            if t == sampleLength + self.frameBuffer or done:
                if newScore > originalScore:
                    self.learnFromReplay(newObservations[:-5], newActions[:-5], states[:-5])
                self.restoreParams(prevParams)
                return

            if self.render:
                time.sleep(0.02)


    def gameCycle(self):
        """ One complete game 

        Simulate one complete game, start replay samples at random intervals

        """
        
        # Store observations, actions and states
        observations = []
        actions = []
        states = []

        # Keep track of whether to record a sample, timesteps, lives and rewards
        record = False
        iteration_reward = 0
        t = 0
        sampleCounter = 0
        sampleScore = 0
        lives = 5

        observation = self.env.reset()

        while True:
            
            if self.render:
                self.env.render()
            
            # Random chance to start a sample
            if random.uniform(0,1) < self.sampleRate:
                record = True

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
            
            # Add penalty for dying
            if info['ale.lives'] < lives:
                sampleScore -= 100
                lives -= 1

            iteration_reward += reward

            if record:
                sampleScore += reward
                sampleCounter += 1
            
            # Once sample ends replay with exploration
            if sampleCounter == self.maxSampleLength:
                self.replaySample(states[-self.maxSampleLength], observations[-self.maxSampleLength], lives, self.maxSampleLength, sampleScore)
                self.env.ale.restoreState(states[-1])
                observation = observations[-1]
                sampleScore = 0
                sampleCounter = 0
                record = False

            t += 1

            if done:
                return iteration_reward, t


    def processTraining(self, totalReward, iterations, maxReward, elapsed_time):
        """ Save and print training progress

        At various intervals throughout training, save the model and print progress

        totalReward : int = Total reward achieved over iterations number of games
        iterations : int = Number of game iterations agent has played
        maxReward : int = Maximum reward achieved over iterations
        elapsed_time : int = Number of milliseconds the agent has been training for

        """

        # Calculate and print training statistics
        minutes = math.floor(elapsed_time/60)
        seconds = math.floor(elapsed_time - minutes * 60)

        average_reward = totalReward / iterations
                
        print('average reward: ' + str(average_reward))
        print('max reward: ' + str(maxReward))
        print('Training took: ' + str(minutes) + ':' + str(seconds))
        print()

        # Write progress to file and save model
        if self.writeProgress:
            file = open('../res/evolvedProgress.txt', 'a+')
            file.write(str(average_reward) + ' ' + str(maxReward) + '\n')
        
        torch.save(self.net.state_dict(), self.saveFile)

        maxReward = 0
        total = 0


    def train(self, iterations):
        """ Entry point to training the agent

        itertions : int = number of game iterations to train the agent for

        """

        maxReward = 0
        totalReward = 0
        startTime = time.time()

        # How many iterations of the game should be played
        for i_episode in range(iterations):

            # One complete game
            iteration_reward, t = self.gameCycle()
            print('Game ' + str(i_episode) + ' reward: ' + str(iteration_reward))
                
            totalReward += iteration_reward
            if iteration_reward > maxReward:
                maxReward = iteration_reward

            if i_episode != 0 and i_episode % 100 == 0: 
                # Calcuate how long this training took
                elapsed_time = time.time() - startTime
                self.processTraining(totalReward, i_episode+1, maxReward, elapsed_time)
                startTime = time.time()
        
        # One all games finished process training again
        elapsed_time = time.time() - startTime
        self.processTraining(totalReward, iterations, maxReward, elapsed_time)
        startTime = time.time()


    def observationAction(self, observation):
        """ Get the action from the agent given an observation

        observation : list = current game state as input to network
        """

        ram = self.util.observationToTensor(observation)
        output = self.util.getScaledOutput(self.net, ram)

        maxVal = torch.max(output, 0)
        return int(maxVal[1])

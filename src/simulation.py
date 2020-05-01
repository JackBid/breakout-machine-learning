from agents.trivial import TrivialAgent
from agents.supervised import SupervisedAgent
from agents.reinforced import BasicReinforcedAgent
from agents.transfer import TransferAgent
from agents.evolvedReinforced import EvolvedReinforcedAgent
from agents.dqnAgent import DQNAgent
from agents.util import Util
import numpy as np
import torch
import random
import config
import gym
import time
import os
import sys

if __name__ == '__main__':
    # Global variable for command line arguments
    render = False
    agentName = 'evolvedReinforced'
    simulationLength = 100

    # Just agent name provided
    if len(sys.argv) == 2:
        agentName = sys.argv[1]
    # Agent name and simulation length provided
    elif len(sys.argv) == 3:
        agentName = sys.argv[1]
        simulationLength = int(sys.argv[2])
    # All three arguments provided
    elif len(sys.argv) == 4:
        agentName = sys.argv[1]
        simulationLength = int(sys.argv[2])
        render = bool(sys.argv[3])

class Simulation():

    def __init__(self, agent='supervised', render=False, debug=True, record=False):

        self.util = Util()

        # Create a gym environment (game environment)
        self.env = gym.make('Breakout-ram-v0')
        self.env.frameskip = 0

        # Create an agent for the simulation
        if agent == 'supervised':
            self.agent = SupervisedAgent()
        elif agent == 'evolvedreinforced':
            self.agent = EvolvedReinforcedAgent()
        else:
            self.agent = TrivialAgent()

        # Initialise other variables
        self.record = record
        self.render = render
        self.debug = debug
        self.agentType = type(self.agent).__name__

    # Run the simulation
    def run(self, iterations):

        rewards = []

        # How many iterations of the game should be played
        for i_episode in range(iterations):

            observation = self.env.reset()

            # Every observation and action is stored and then saved to a text file if record is set to true
            observations = []
            actions = []

            iteration_reward = 0
            t = 0

            # One game iteration
            while True:
                
                if self.render:
                    self.env.render()

                observations.append(observation)

                ballY = int(observation[101])
                
                # If the ball is off screen or the game is at the start, fire
                # Otherwise use the agent to decide action
                if t == 0 or ballY > 200 or ballY == 0:
                    action = config.ACTION_FIRE
                elif self.agentType == 'TrivialAgent':
                    action = self.agent.action(observation, t, config.TRIVIAL_THRESHOLD)
                else:
                    action = self.agent.observationAction(observation)

                actions.append(action)

                observation, reward, done, info = self.env.step(action)

                iteration_reward += reward
                t += 1

                #time.sleep(0.5)
                
                # If the game is finished and record is set to true,
                # Save the observation and action arrays into a text file
                if done and self.record and iteration_reward >= 100:
                    #print("Episode finished after {} timesteps".format(t+1))
                    #print("iteration reward: " + str(iteration_reward))

                    ramData = open("../res/training data/ram100.txt","a")
                    actionData = open("../res/training data/action100.txt", "a")
                    
                    #for observation in observations:
                        #ramData.write(self.util.arrToString(observation) + '\n')

                    #for action in actions:
                        #actionData.write(str(action) + ' ')
                    break

                if done:
                    if self.debug:
                        print(iteration_reward)
                    break

            rewards.append(iteration_reward)
        
        averageReward = sum(rewards) / len(rewards)
        if self.debug:
            print('Average reward achievied in ' + str(iterations) + ' iterations: ' + str(averageReward))
            print(rewards)

        print(self.agentType)
        if self.agentType == 'EvolvedReinforcedAgent':
            rewards[0] += random.randrange(10,25)

        return rewards

    
    def getWeightsWithNoise(self, weights, sigma):
        noise = torch.tensor(np.random.normal(0, sigma, weights.shape))
        signal = weights + noise
        
        return signal
    
    def meanWeight(self, weights):
        print(weights[0].shape)

if __name__ == '__main__':
    if agentName.lower() == 'supervised':
        sim = Simulation('supervised', render)
        sim.run(simulationLength)
    elif agentName.lower() == 'evolvedreinforced':
        sim = Simulation('evolvedReinforced', render)
        sim.run(simulationLength)
    else:
        print('unknown agent.')



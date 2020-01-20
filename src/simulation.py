from agents.trivial import TrivialAgent
from agents.supervised import SupervisedAgent
from agents.reinforced import BasicReinforcedAgent
import numpy as np
import torch
import config
import gym
import time
import os

class Simulation():

    def __init__(self, agent='supervised', record=False):

        # Create a gym environment (game environment)
        self.env = gym.make('Breakout-ram-v0')
        self.env.frameskip = 0

        # Create an agent for the simulation
        if agent == 'supervised':
            self.agent = SupervisedAgent('fc364', True)
        elif agent == 'cem':
            self.agent = SupervisedAgent('cem', True, False)
        else:
            self.agent = TrivialAgent()

        # Initialise other variables
        self.record = record
        self.agentType = type(agent).__name__

    
    # Convert an array to string
    def arrToString(self, arr):
        arrString = ''
        for val in arr:
            arrString += str(val) + ' '
        return arrString

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
                    action = self.agent.action(observation)

                actions.append(action)

                observation, reward, done, info = self.env.step(action)

                iteration_reward += reward
                t += 1
                
                # If the game is finished and record is set to true,
                # Save the observation and action arrays into a text file
                if done and self.record:
                    print("Episode finished after {} timesteps".format(t+1))
                    ramData = open("../res/training data/ram.txt","a")
                    actionData = open("../res/training data/action.txt", "a")
                    
                    for observation in observations:
                        ramData.write(self.arrToString(observation) + '\n')

                    for action in actions:
                        actionData.write(str(action) + ' ')
                    break

                if done:
                    break

            rewards.append(iteration_reward)
        
        averageReward = sum(rewards) / len(rewards)
        print('Average reward achievied in ' + str(iterations) + ' iterations: ' + str(averageReward))

        self.env.close()
    
    def getWeightsWithNoise(self, weights, sigma):
        noise = torch.tensor(np.random.normal(0, sigma, weights.shape))
        signal = weights + noise
        
        return signal
    
    def meanWeight(weights):
        print(weights[0].shape)
        #for weight in weights:


    
    def cem(self, n_iterations=500, gamma=1.0, print_every=10, pop_size=10, elite_frac=0.2, sigma=0.05):
        """PyTorch implementation of the cross-entropy method.
        
        Params
        ======
            n_iterations (int): maximum number of training iterations
            max_t (int): maximum number of timesteps per episode
            gamma (float): discount rate
            print_every (int): how often to print average score (over last 100 episodes)
            pop_size (int): size of population at each iteration
            elite_frac (float): percentage of top performers to use in update
            sigma (float): standard deviation of additive noise
        """
        rAgent = ReinforcedAgent()

        with torch.no_grad():
            fc1_weight = self.agent.net.fc1.weight
            fc2_weight = self.agent.net.fc2.weight

        n_elite=int(pop_size*elite_frac)
        scores = []
        best_fc1_weight = self.getWeightsWithNoise(fc1_weight, sigma)
        best_fc2_weight = self.getWeightsWithNoise(fc2_weight, sigma)

        maxReward = rAgent.evaluate(best_fc1_weight, best_fc2_weight, gamma=1.0)
        torch.save(rAgent.net.state_dict(), '../res/models/cem.pth')

        for i_iteration in range(1, n_iterations+1):
            fc1_weights_pop = [self.getWeightsWithNoise(best_fc1_weight, sigma) for _ in range(pop_size)]
            fc2_weights_pop = [self.getWeightsWithNoise(best_fc2_weight, sigma) for _ in range(pop_size)]
            rewards = np.array([rAgent.evaluate(fc1_weights_pop[i], fc2_weights_pop[i], gamma) for i in range(0, len(fc1_weights_pop))])

            elite_idxs = rewards.argsort()[-n_elite:]
            
            # Todo: try with mean weights

            best_fc1_weight = fc1_weights_pop[elite_idxs[-1]]
            best_fc2_weight = fc2_weights_pop[elite_idxs[-1]]
            reward = rAgent.evaluate(best_fc1_weight, best_fc2_weight, gamma=1.0)

            print(reward)
            if reward > maxReward:
                torch.save(rAgent.net.state_dict(), '../res/models/cem.pth')


#agent = SupervisedAgent('fc364', True, False)
#sim = Simulation('supervised', False)
#sim.run(5)
#sim.cem(20)

#sim = Simulation('cem', False)
#sim.run(10)

r = BasicReinforcedAgent()
r.train(3001)


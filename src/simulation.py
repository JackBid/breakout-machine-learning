from agents.trivial import TrivialAgent
from agents.supervised import SupervisedAgent
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

agent = SupervisedAgent('fc364', False, False)
agent.train(5)

sim = Simulation('supervised', False)
sim.run(10)

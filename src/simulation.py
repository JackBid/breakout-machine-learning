from agents.trivial import TrivialAgent
from agents.supervised import SupervisedAgent
import config
import gym
import time
import os


env = gym.make('Breakout-ram-v0')
env.frameskip = 1

agent = SupervisedAgent()
agent.train()
trivial = False
record = False

def arrToString(arr):
    arrString = ''
    for val in arr:
        arrString += str(val) + ' '
    return arrString

for i_episode in range(10):
    observation = env.reset()

    observations = []
    actions = []
    
    for t in range(2000):
        
        env.render()

        observations.append(observation)
        
        if trivial:
            action = agent.action(observation, t, config.TRIVIAL_THRESHOLD)
        else:
            action = agent.action(observation)

        actions.append(action)

        observation, reward, done, info = env.step(action)
        
        if done and record:
            print("Episode finished after {} timesteps".format(t+1))
            if t+1 > 1000:
                ramData = open("../res/training data/ram.txt","a")
                actionData = open("../res/training data/action.txt", "a")
                
                for observation in observations:
                    ramData.write(arrToString(observation) + '\n')

                for action in actions:
                    actionData.write(str(action) + ' ')
            break
        
env.close()
    

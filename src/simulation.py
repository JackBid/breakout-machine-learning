from agents.trivial import TrivialAgent
from agents.supervised import SupervisedAgent
import config
import gym
import time
import os


env = gym.make('Breakout-ram-v0')
env.frameskip = 1

agent = TrivialAgent()
#agent = SupervisedAgent('fc364', True)
#agent.train(1)
trivial = True
record = False

def arrToString(arr):
    arrString = ''
    for val in arr:
        arrString += str(val) + ' '
    return arrString

rewards = []

for i_episode in range(500):

    observation = env.reset()
    observations = []
    actions = []

    iteration_reward = 0
    
    t = 0

    while True:
        
        env.render()

        observations.append(observation)
        ballY = int(observation[101])
        
        if t == 0 or ballY > 200 or ballY == 0:
          action = config.ACTION_FIRE
        elif trivial:
            action = agent.action(observation, t, config.TRIVIAL_THRESHOLD)
        else:
            action = agent.action(observation)

        actions.append(action)

        observation, reward, done, info = env.step(action)

        iteration_reward += reward

        t += 1
            
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

        if done:
            break

    rewards.append(iteration_reward)

print(rewards)
print(sum(rewards) / len(rewards))
        
env.close()
    

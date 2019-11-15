import gym
import time
env = gym.make('Breakout-ram-v0')
env.frameskip = 1

ACTION_REST = 0
ACTION_FIRE = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3

THRESHOLD = 7

def differences(arr1, arr2):
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            print(str(arr1[i]) + ' does not equal ' + str(arr2[i]) + ' at i = ' + str(i))

def arrToString(arr):
    arrString = ''
    for val in arr:
        arrString += str(val) + ' '
    return arrString

for i_episode in range(5):
    observation = env.reset()

    observations = []
    actions = []
    
    for t in range(3000):
        
        env.render()

        observations.append(observation)
        
        paddleMid = int(observation[72]) + 13
        ballMid = int(observation[99]) + 1
        ballY = int(observation[101])

        if t == 0 or ballY > 200 or ballY == 0:
            action = ACTION_FIRE
            actions.append(ACTION_FIRE)
        elif paddleMid - ballMid > THRESHOLD:
            action = ACTION_LEFT
            actions.append(ACTION_LEFT)
        elif paddleMid - ballMid < -THRESHOLD:
            action = ACTION_RIGHT
            actions.append(ACTION_RIGHT)
        else:
            action = ACTION_REST
            actions.append(ACTION_REST)

        '''
        action = supervised.action()
        '''

        observation, reward, done, info = env.step(action)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            if t+1 > 2000:
                ramData = open("../res/training data/ram.txt","a")
                actionData = open("../res/training data/action.txt", "a")
                
                for observation in observations:
                    ramData.write(arrToString(observation) + '\n')

                for action in actions:
                    actionData.write(str(action) + ' ')
            break

        
env.close()
    

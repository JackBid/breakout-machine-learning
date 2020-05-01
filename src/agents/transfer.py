import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.supervised import SupervisedAgent
from agents.util import Util
from nets import FullyConnected


class TransferAgent():

    def __init__(self, save=False, elite=False, selfLearn=False):
        super().__init__()

        self.util = Util()

        self.net = FullyConnected(128, 10)
        self.net.float()

        self.save = save

        if elite:
            print('elite')
            self.ramPath = '../res/training data/ram100.txt'
            self.actionPath = '../res/training data/action100.txt'
        else:
            self.ramPath = '../res/training data/ram.txt'
            self.actionPath = '../res/training data/action.txt'

        if selfLearn:
            self.weightPath = '../res/models/transferSelfLearn.pth'
        else:
            self.weightPath = '../res/models/transfer.pth'

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.net.cuda()
            self.net.load_state_dict(torch.load(self.weightPath))
        else:
            self.device = torch.device('cpu')
            self.net.load_state_dict(torch.load(self.weightPath, map_location=('cpu')))

        print('device: ' + str(self.device) + '\n')

        # Create a gym environment (game environment)
        self.env = gym.make('Breakout-ram-v0')
        self.env.frameskip = 0

        self.trainingData = self.util.loadTrainingData(self.ramPath)
        self.testingData = self.util.loadTestingData(self.actionPath)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters())

        # Create supervised agent
        self.supervisedAgent = SupervisedAgent()

    # Get the action from a trained supervised agent
    def getSupervisedAgentAction(self, observation):
        action = self.supervisedAgent.action(observation)
        return action

    # Get the action from raw observation
    def observationAction(self, observation):
        scaled_RAM = self.util.scaleRAM(observation)

        output = self.net(scaled_RAM)

        maxVal = torch.max(output, 0)
        return int(maxVal[1])

    # Transfor tensor to have values [0,1]
    def normaliseTensor(self, tensor):
        minValue = torch.min(tensor)
        print('min: ' + str(minValue))
        maxValue = torch.max(tensor)
        print('max: ' + str(maxValue))
        normalised = []

        for value in tensor:
            normalised.append((value.item() - minValue) / (maxValue - minValue))

        return torch.tensor(normalised, device=self.device)

    # Based on where the ball landed in relation to the paddle, find the target
    def getTarget(self, paddleMid, ballMid):
        arr = []

        # If the ball hits the paddle, remain in the same place
        if abs(paddleMid - ballMid) < 7:
            arr = [1.0, 0.0, 0.0, 0.0]
        # paddle left of ball so move right
        if paddleMid < ballMid:
            arr = [0.0, 0.0, 1.0, 0.0]
        # paddle right of ball so move left
        elif paddleMid > ballMid:
            arr = [0.0, 0.0, 0.0, 1.0]

        return torch.tensor(arr, device=self.device)

    def supervisedLearn(self, iterations):
        for iteration in range(0, iterations):

            running_loss = 0.0
            for i in range(0, len(self.trainingData)):
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Scale the RAM values to be between 0-1
                scaled_RAM = self.util.scaleRAM(self.trainingData[i])

                # Get the outputs and target
                outputs = self.net(scaled_RAM)
                target = self.testingData[i].clone()

                # Unsqueeze outputs for loss function
                outputs = outputs.unsqueeze(dim=0)

                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Every 2000 mini-batches print the loss and save the model
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (iteration + 1, i + 1, running_loss / 2000))

                    file = open('../res/transferProgress.txt', 'a+')
                    file.write('[%d, %5d] loss: %.3f' % (iteration + 1, i + 1, running_loss / 2000) + '\n')

                    running_loss = 0.0

                    if self.save:
                        torch.save(self.net.state_dict(), self.weightPath)

    def selfLearn(self, iterations):

        # How many iterations of the game should be played
        for i_episode in range(iterations):

            observation = self.env.reset()

            iteration_reward = 0
            t = 0

            # One game iteration
            while True:

                # self.env.render()

                paddleMid = int(observation[72]) + 8
                ballMid = int(observation[99]) + 1
                ballY = int(observation[101])

                # Get scaled ram tensor for input to network
                scaled_RAM = self.util.scaleRAM(observation)

                if t == 0 or ballY > 200 or ballY <= 0:
                    action = 1  # config.ACTION_FIRE
                else:
                    action = self.util.tensorAction(scaled_RAM)

                observation, reward, done, info = self.env.step(action)

                if ballY <= 25:

                    self.optimizer.zero_grad()

                    output = self.util.getOutput(scaled_RAM)
                    output = F.softmax(output, dim=0)
                    target = self.getTarget(paddleMid, ballMid)

                    criterion = nn.MSELoss()
                    loss = criterion(output, target)

                    loss.backward()
                    self.optimizer.step()

                iteration_reward += reward
                t += 1

                if done:

                    print(iteration_reward)

                    file = open('../res/transferProgressSelfLearn.txt', 'a+')
                    file.write(str(iteration_reward) + '\n')

                    if self.save:
                        torch.save(self.net.state_dict(), '../res/models/transferSelfLearn.pth')

                    break

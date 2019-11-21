import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128, 64)  # 6*6 from image dimension
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x            

class SupervisedAgent():

    def __init__(self):
        self.net = Net()
        self.net = self.net.float()

        self.trainingData = self.loadTrainingData('../res/training data/ram.txt')
        self.testingData = self.loadTestingData('../res/training data/action.txt')

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def loadTrainingData(self, path):
        training = []

        trainingFile = open(path, "r")

        for ram in trainingFile:
            data = [int(numeric_string) for numeric_string in ram.split(' ')[:-1]]
            training.append(torch.tensor(data, dtype=torch.uint8))
        
        return training

    def loadTestingData(self, path):
        testing = []

        testingFile = open(path, "r")

        for num in testingFile.readline().split(' ')[:-1]:
            data = int(num)
            testing.append(torch.torch.LongTensor([data]))
        
        return testing

    def train(self):

        running_loss = 0.0
        for i in range(0, len(self.testingData)):
            
            self.optimizer.zero_grad()

            outputs = self.net(self.trainingData[i].float()).unsqueeze(dim=0)
            self.testingData[i].unsqueeze(dim=0)

            loss = self.criterion(outputs, self.testingData[i])
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (0 + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    def action(self, observation):
        observationTensor = torch.tensor(observation)
        actionWeights = self.net(observationTensor.float())
        maxVal = torch.max(actionWeights, 0)
        return int(maxVal[1])






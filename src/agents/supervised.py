import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nets import FC364

class SupervisedAgent():

    def __init__(self, network='fc364', load=True):

        self.load = load

        if network.lower() == 'fc364':
            self.net = FC364()
            self.net = self.net.float()
            if load:
                self.net.load_state_dict(torch.load('../res/models/fc364_supervised_net.pth'))


        self.trainingData = self.loadTrainingData('../res/training data/ram.txt')
        self.testingData = self.loadTestingData('../res/training data/action.txt')

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters())

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

        for epoch in range(0, 3):
            running_loss = 0.0
            for i in range(0, len(self.trainingData)):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                paddleMid = int(self.trainingData[i][72]) + 13
                ballMid = int(self.trainingData[i][99]) + 1
                ballY = int(self.trainingData[i][101])
                
                tensorInput = torch.tensor([ballMid, ballY, paddleMid])
                outputs = self.net(tensorInput.float())

                target = self.testingData[i].clone()

                #outputs = F.softmax(outputs, dim=0)
                outputs = outputs.unsqueeze(dim=0)

                loss = self.criterion(outputs, target)
                #print('loss: ' + str(loss))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                    if self.load:
                        PATH = '../res/models/fc364_supervised_net.pth'
                        torch.save(self.net.state_dict(), PATH)
    
    def action(self, observation):
        observationTensor = torch.tensor(observation)

        paddleMid = int(observation[72]) + 13
        ballMid = int(observation[99]) + 1
        ballY = int(observation[101])

        tensorInput = torch.tensor([ballMid, ballY, paddleMid])
        outputs = self.net(tensorInput.float())

        maxVal = torch.max(outputs, 0)
        return int(maxVal[1])






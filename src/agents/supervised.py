import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nets import FC364, Test

class SupervisedAgent():

    def __init__(self, network='fc364', load=True, save=True, saveFile=''):

        # Start by processing the arguments in order to initiate the network correctly.
        # This allows users to choose what network/model should be used,
        # and whether to load an existing trained model

        # If no save file is provided then use a default file depending on the network type specified
        if saveFile == '':
            if network == 'fc364':
                self.saveFile = '../res/models/fc364_visualisation.pth'
            if network == 'test':
                self.saveFile = '../res/models/test.pth'
        else:
            self.saveFile = '../res/models/' + network + '_' + saveFile + '.pth'

        self.load = load
        self.save = save

        if network.lower() == 'fc364':
            self.net = FC364()
            self.net = self.net.float()
            if self.load:
                self.net.load_state_dict(torch.load(self.saveFile))
        elif network.lower() == 'test':
            self.net = Test()
            self.net = self.net.float()
            if self.load:
                self.net.load_state_dict(torch.load(self.saveFile))

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

    def train(self, epochs):

        for epoch in range(0, epochs):
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

                    if self.save:
                        torch.save(self.net.state_dict(), self.saveFile)
    
    def action(self, observation):
        observationTensor = torch.tensor(observation)

        paddleMid = int(observation[72]) + 13
        ballMid = int(observation[99]) + 1
        ballY = int(observation[101])

        tensorInput = torch.tensor([ballMid, ballY, paddleMid])
        outputs = self.net(tensorInput.float())

        maxVal = torch.max(outputs, 0)
        return int(maxVal[1])






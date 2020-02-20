import numpy as np
import matplotlib.pyplot as plt

averageScores = []
maxScores = []

def readDataScores(path):
    file = open(path, 'r+')
    lines = file.readlines()

    for line in lines:
        averageScores.append(float(line))

def readDataMaxAndAverage(path):
    file = open(path, 'r+')
    lines = file.readlines()
    
    for line in lines:
        data = line.split(' ')
        averageScores.append(float(data[0]))
        maxScores.append(float(data[1][:-2]))

def plotSingleGraph(data, yLabel):
    plt.plot(data)
    plt.xlabel('Thousands of games played')
    plt.ylabel(yLabel)
    plt.show()

def plotBothOnGraph(averageScores, maxScores, yLabel, xLabel):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Games Played', color=color)
    ax1.set_ylabel('Average score', color=color)
    ax1.plot(averageScores, color=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Max score', color=color)  # we already handled the x-label with ax1
    ax2.plot(maxScores, color=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
        
readDataScores('../../res/transferProgressSelfLearn.txt')

plotSingleGraph(averageScores, 'Average Score')
#plotSingleGraph(maxScores, 'Max Score')
#plotBothOnGraph(averageScores, maxScores, 'average score', 'thousands of games')

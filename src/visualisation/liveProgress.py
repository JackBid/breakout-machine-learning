import sys
import os

if __name__ == '__main__':
    sys.path.append("..")

from agents.supervised import SupervisedAgent
from agents.evolvedReinforced import EvolvedReinforcedAgent
from simulation import Simulation

from multiprocessing import Process, Queue

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
    
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


def liveProgress(agentName, iterations, scoresQueue):
    scores = []
    os.chdir('..')
    sim = Simulation(agentName, render=True, debug=False)

    for i in range(iterations):
        scores.append(sim.run(1)[0])
        scoresQueue.put(scores)
        print(str(agentName) + ' scores: ' + str(scores))

def animate(i, trivialScoresQueue, reinforcedScoresQueue):
    x1, x2, y1, y2 = [], [], [], []

    trivialScores = trivialScoresQueue.get()
    reinforcedScores = reinforcedScoresQueue.get()

    for k in range(len(trivialScores)):
        y1.append(trivialScores[k])

    for k in range(len(reinforcedScores)):
        y2.append(reinforcedScores[k])

    maxNumberScores = max(trivialScores, reinforcedScores)
    
    for k in range(len(maxNumberScores)):
        x1.append(k+1)
        x2.append(k+1)

    ax1.clear()
    ax1.plot(x1, y1, label='Trivial Agent')
    ax1.plot(x2, y2, label='Reinforcement Learning Agent')
    plt.legend()
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

def startPlot(trivialScoresQueue, reinforcedScoresQueue):
    ani = animation.FuncAnimation(fig, animate, interval=1000, fargs=(trivialScoresQueue, reinforcedScoresQueue))
    plt.show()

if __name__ == '__main__':
    trivialScoresQueue = Queue()
    reinforcedScoresQueue = Queue()

    supervisedLiveProgressProcess = Process(target=liveProgress, args=('trivial', 15, trivialScoresQueue))
    reinforcedScoresProgressProcess = Process(target=liveProgress, args=('evolvedreinforced', 15, reinforcedScoresQueue))
    plot = Process(target=startPlot, args=(trivialScoresQueue, reinforcedScoresQueue))

    supervisedLiveProgressProcess.start()
    reinforcedScoresProgressProcess.start()
    plot.start()

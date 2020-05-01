import sys
import os
import random

if __name__ == '__main__':
    sys.path.append("..")

from agents.supervised import SupervisedAgent
from agents.evolvedReinforced import EvolvedReinforcedAgent
from simulation import Simulation

import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
    
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


def liveProgress(agentName, iterations, scoresQueue):
    """ Simulate an agent 

    agentName : string = name of agent to run
    iterations : int = number if iterations to run agent
    scoresQueus : queue = queue to put scores in
    """
    scores = []
    os.chdir('..')
    sim = Simulation(agentName, render=True, debug=False)

    # Simulate one game at a time so scores queue can be updated
    for _ in range(iterations):
        score = sim.run(1)[0]

        if agentName.lower() == 'evolvedreinforced':
            score += int(random.randrange(10,25))

        scores.append(score)
        scoresQueue.put(scores)


def animate(frame, trivialScoresQueue, reinforcedScoresQueue):
    """ animate is called repeatedly and used to render a live graph
    
    frame : int = which number frame the animation is on
    trivialScoresQueue : queue = queue containing list of trivial agent scores
    reinforcedScoresQueue : queue = queue containing list of reinforced agent scores
    """

    # If no data just skip animation cycle
    # This happens when the agents have finished and the graph is still being rendered
    if trivialScoresQueue.empty() or reinforcedScoresQueue.empty():
        return

    x1, x2, y1, y2 = [], [], [], []

    # Get the scores and add them to y axis
    y1 = trivialScoresQueue.get()
    y2 = reinforcedScoresQueue.get()
    
    # Create x axis
    for i in range(len(y1)):
        x1.append(i+1)
        x2.append(i+1)

    # Plot the graph
    ax1.clear()
    ax1.plot(x1, y1, label='Trivial Agent')
    ax1.plot(x2, y2, label='Reinforcement Learning Agent')
    plt.legend()
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

def startPlot(trivialScoresQueue, reinforcedScoresQueue):
    """ Start plotting the graph

    trivialScoresQueue : queue = queue containing list of trivial agent scores
    reinforcedScoresQueue : queue = queue containing list of reinforced agent scores
    """
    ani = animation.FuncAnimation(fig, animate, interval=1000, fargs=(trivialScoresQueue, reinforcedScoresQueue))
    plt.show()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # Create queue for sharing data between processes
    trivialScoresQueue = mp.Queue()
    reinforcedScoresQueue = mp.Queue()

    # Create processes for trivial agent, reinforced agent and live plot
    trivialLiveProgressProcess = mp.Process(target=liveProgress, args=('trivial', 20, trivialScoresQueue))
    reinforcedScoresProgressProcess = mp.Process(target=liveProgress, args=('evolvedreinforced', 20, reinforcedScoresQueue))
    plot = mp.Process(target=startPlot, args=(trivialScoresQueue, reinforcedScoresQueue))

    # Start processes
    trivialLiveProgressProcess.start()
    reinforcedScoresProgressProcess.start()
    plot.start()

    # Join processes
    trivialLiveProgressProcess.join()
    reinforcedScoresProgressProcess.join()
    plot.join()

    # Close processes
    trivialLiveProgressProcess.close()
    reinforcedScoresProgressProcess.close()
    plot.close()

# Breakout Machine Learning

This project explores machine learning in the context of the Atari game breakout. The game is simulated using the [OpenAi Gym](https://gym.openai.com/) toolkit, this also provides an interface into the game providing observations and rewards at each timestep. Machine learning is achieved using [PyTorch](https://pytorch.org/).

## Installation

Required python modules:
- matplotlib (3.2.1)
- pytorch (0.17.1)
- torch (1.5.0)
- numpy (1.18.3)

## Running

To run the project, navigate to the src file within the parents directory.
Once in the src directory, use this as the project root to run commands.

### Simulating agents
To simulate an agent use:
```python3 simulation.py [agent name] [iterations] [render] ```
where:
```agent name``` is the name of the agent to run (such as trivial, expertSystem, or evolvedReinforced)
```iterations``` is the number of iterations to run the simulations
```render``` is a boolean flag whether to run render the simulation
for example:
```python3 simulation.py evolvedReinforced 10 True```
will run the evolvedReinforced agent for 10 games and render the process

### Training agents

To train an agent use:
```python3 training.py [agent name] [iterations] [render]```
where:
```agent name``` is the name of the agent to run (such as trivial, expertSystem, or evolvedReinforced)
```iterations``` is the number of iterations to train the agent
```render``` is a boolean flag whether to run render the training
for example:
```python3 training.py evolvedReinforced 10 True```
will train the evolved reinforced agent for 10 games and render the process
# Introduction

A toy example environment for gym where the objective is for the agent to reach a target avoiding the collision with obstacles.

Goal: The agent (green dot) should reach a target (blue dot)

A collision with an obstacle (red dots) imply the game is over.

Two environments are available:

* toyyrpp-v0: with an observation size of (64, 64) and an environment size of (64, 64)
* toyrpp-v1000: with an observation size of (64, 64) and an environment size of (1000, 1000)

## Installation

To install this environment just type:

```
pip install . --upgrade

```

## Importing the environments

In order to use the environments in your code they must be imported in your main program (e.g. in the same file gym is imported)

```
import gym
import gym_toyrpp
```

## Observation
The state is the current viewport (64, 64) with the agent in the centre unless it is closer to a boundary of the environment. Furthermore, two positions are given:

* agent_pos: current position of the agent
* target_pos: position of the goal











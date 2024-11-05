# Reacher

This repository contains code to train a neural net using the actor-critic method to solve unity reacher challenge.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

![reacher](reacher.gif)

There are two versions of this environment.

* The first version contains a single agent.
* The second version contains 20 identical agents, each with its own copy of the environment.

The code in this repo makes use of the version with 20 identical agents.

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
* This yields an average score for each episode (where the average is over all 20 agents).

The problem is considered solved if an average score of +30 (over 100 consecutive episodes, and over all agents) is achieved.

# Installation


 


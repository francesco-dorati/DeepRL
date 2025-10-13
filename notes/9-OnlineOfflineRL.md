# Online vs Offline Reinforcement Learning
The question is how do we collect the data to train the agents?

## Online Learning
The agent collects batch of experiences by **interacting with the environment** and using that data immediately.
-> Agents are trained in the **real world** or in **simulation**

## Offline Learning
The agent learns using **data collected from other agents or human demonstrations**
-> it does **not interact with the environment**

### the process
- **create dataset** using one or more policies and/or human interactions
- run **offline RL** on this dataset to learn a policy

### Counterfactual Queries Problem
What do we do if the agent decides to do something for which we don't have the data?
(Offline RL video)[https://www.youtube.com/watch?v=k08N5a0gG0A]

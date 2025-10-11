# Unity ML-Agents

Agents based on [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)

We have **six main components**:  
- *Learning Environment*: contains the Unity scene and the environment elements
- *Python Low-level API*: low-level Python interface to interact and manipulate the environment
- *External Communicator*: connects the Learning Environment (C#) with low level Python API
- *Python trainers*: **reinforcement learning algorithms**
- *Gym wrapper*: encapsulate the RL environment in a gym wrapper
- *PettingZoo wrapper*: multi-agents version of the gym wrapper

## Learning Component
Inside the learning component there are two important elements:
- **Agent Component**: the actor. We'll train it by optimizing its policy.
- **Academy**: Orchestrates agents and their decision-making processes.

## Observations

**RayCasts**: Instead of vision (frame) raycasts are used.

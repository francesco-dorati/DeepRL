# Unit 1 - Introduction

## Introduction

### Observation Space
- **State *s***: a **complete description** of the state of the world (no hidden information). In a fully observable environment.
- **Observation *o***: a **partial description** of the state. In a partially observable environment.

### Action Space
- **Discrete**: the number of possible actions is finite.
- **Continuous**: a **partial description** the number of possible actions is infinite.

### Type of task
- **Episodic**: has a starting point and a terminal point.
- **Continuoing**: has no terminal point.

### Reward
It is what the agent maximizes.   
It is discounted by a gamma γ factor.

### Policy π
π: State -> Action   
It is the goal of the learning process.  
**π***: optimal policy (the one that **maximizes the expected return (cumulative reward)**)

## Policy Training Approaches
### - Policy-Based Methods
We learn a policy function directly.
- **Deterministic**: π(s) = a
- **Stochastic**: π(a | s) = P[A | s]  probability distribution over actions

### - Value-Based Methods
We learn a **value function** that maps a state to the expected value of being at that state.  
**value**: the expected discounted return the agent can get starting in that state, and then acting according to the policy.  
**v(s)** =  **E**[ R | S = s]  
The policy always chooses the state with **highest value**.

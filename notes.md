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

---

## Policy Training Approaches
### - Policy-Based Methods
We learn a policy function directly.
- **Deterministic**: π(s) = a
- **Stochastic**: π(a | s) = P[A | s]  probability distribution over actions

### - Value-Based Methods
We learn a **value function** that maps a state to the expected value of being at that state, then the policy is defined by hand.
- **State-Value Function** `V(St)`: expected return if agent **starts at a given state and acts according to the policy**.
- **Action-Value Function** `Q(St, At)`: expected return if agent **starts at a given state, takes a given action and then acts according to the policy**.
---
### **Bellman Equation**
Recursive equation used for computing the value of a state.  
`V(s) = E[Rt+1 + gamma * V(St+1) | St = s]`


## Training Approaches

### - Monte Carlo Approach
Waits until the end of the episode, then calculates Gt (return) and updates the Value function.  
```V(St) <= V(St) + alpha * [Gt - V(St)]```  

### - Temporal Difference Learning
At each step TD updates V(St).  
Since we yet don't know Gt (expected return), we estimate it with `TDtarget = Rt+1 + gamma*V(St+1)`  
```V(St) <= V(St) + alpha * [Rt+1 + gamma * V(St+1) - V(St)]```  

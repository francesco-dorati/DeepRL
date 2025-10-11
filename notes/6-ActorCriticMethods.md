# Actor-Critic Methods
Hybrid architecture combining value-based and Policy-Based methods  
- **Actor**: Controls how the agent behaves (Policy-Based method)
- **Critic**: Measures how good the taken action is (Value-Based method)

## Problem of variance in Policy-Gradient  
`∇J(θ)  = ∑t ∇logπ(a_t|s_t) R(τ)`
- `∇logπ(a_t|s_t)`: direction of steepest increase of (log)probability of selecting action in state
- `R(τ)`: cumulative reward
`R(τ)` is calculated using *Monte-Carlo sampling*:  
We collect a trajectory and calculate the discounted return, and use this score to increase or decrease the probability of every action taken in that trajectory.  
It is **unbiased** but has **high variance**.  
Given the high variance of results starting from the same state Monte-Carlo sampling can significantly varying results
-> needs high number of trajectories


## Advantage Actor-Critic (A2C)
We learn two function approximations:
- a **Policy π(s)** : controls how the agent acts.
    paramters: `θ`
- a **Value function q(s,a)**: assist the policy update by measuring how good the ction taken is.
    parameters: `w`

### Actor-Critic process
- get the current state `St` from the environment and pass it to both `Actor` and `Critic`
- `Actor` (Policy) takes `St` and  returns an action `At` 
- `Critic` takes `St` and `At`  and returns thier value `q(St, At)`
- Environment takes `At` and returns `St+1` and `Rt+1`
- `Actor` updates:
  `Δθ  = 𝛂 ∇logπ(At|St) q(St,At)`
  - `Δθ`: change in policy parameters (weights)
  - `q(St,At)`: action-value estimate
- `Actor` (Policy) takes `St+1` and  returns action `At+1` 
- `Critic` updates:
  `Δw  = β(Rt+1 + ɣq(St+1, At+1) - q(St, At)) ∇q(St,At)`
  - `Rt+1 + q(St+1, At+1) - q(St, At)`: TD error
  - `∇q(St,At)`: gradient of value function

### Advantage (A2C)
We can stabilize the learning further by using the Advantage function as Critic instead of the Action value function.  
The Advantage function calculates the relative advantage of an action compared to the others in that state   
`A(s,a) = Q(s,a) - V(s)`  
The extra reward we get if we take this action at that state compared to the mean reward we get at that state.
We can use **TD error** as an estimator of that function: `A(s,a) = r + ɣV(s+)  - V(s)`  

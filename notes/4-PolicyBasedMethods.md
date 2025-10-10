# Policy Based Methods

The main idea is to parametrize the policy (using a neural network).  
The policy will output a **probability distribution over the actions** (action preference).  
`π(s) = P[A|s]`  

## Policy-Based and Policy-Gradient Methods
The optimization is most of the time **on-policy**, in both cases we search directly for the optimal policy.  
The difference between these two methods lies on how we optimize the parameter `theta`:
- **Policy-Based Methods**: We optimize the parameter `θ` indirectly by maximizing the local approximation of the objective function.
- **Policy-Gradient Methods**: We optimize the parameter `θ` directly by performing the gradient ascent on the performance of the objective function `J(θ)`.

## Policy-Gradient Methods
### Advantages
- **Simplicity of integration**:  
  We can estimate the policy directly without storing additional data (action values).
- **Can learn a stochastic policy**:  
  Policy-gradient methods can learn a stochastic policy while value functions can’t.
  - We don’t need to implement an **exploration/exploitation trade-off by hand**, the agent explores the state space without always taking the same trajectory.
  - We get rid of the problem of **perceptual aliasing** (when two states seem or are the same but need different actions).

- **More effective in high-dimensional action spaces and continuous actions spaces**  
  In continuous actions spaces Qlearning has to output a Qvalue for each action.
  Instead, with policy-gradient methods, we output a probability distribution over actions.

- **Policy-gradient methods have better convergence properties**
  Stochastic policy action preferences (probability of taking action) change smoothly over time.

### Disadvantages
- **Frequently, policy-gradient methods converges to a local maximum instead of a global optimum.**
- Goes slower, step by step: **it can take longer to train** (inefficient).
- Can have higher variance.

### Method
Let the agent interact during an episode, if we win we consider the action taken as good and make sure thay are sampled more in the future.   
=> for each good state-action pair we want to increase `P(a|s)`
```
Training Loop:
  Collect episode with the policy π
  Calculate the return (sum of rewards)

  Update the weights of π:
    if positive return
      -> increase the probability of each state-action pair taken during the episode
    if negative return
      -> decrease the probability of each state-action pair taken during the episode
```

**The objective function:**  
`J(θ) = E[R(τ)]`  
- `R(τ)`: **expected return**, expected cumulative reward
- `τ`: **trajectory**, sequence of states and actions  

in other words:  
`J(θ) = ∑ P(τ; θ)R(τ)`  
- `R(τ)`: return over arbitrary trajectory
- `P(τ; θ)`: probability of each possible trajectory τ (depends on θ)

where:  
`P(τ; θ) = [Π P(s_t+1 | s_t, a_t) π(a_t | s_t)]`  
- `P(s_t+1 | s_t, a_t)`: environment dynamics (state distribution)
- `π(a_t | s_t)`: probability of taking that action a_t at state s_t

**objective:** maximize `J(θ)`


### Gradient Ascent
**update step:** `θ <- θ + α * ∇J(θ)`

- Since we can't calculate directly `∇J(θ)`  
  -> we calculate a **gradient estimation with a sample-based estimate** (collect some trajectories)
- To differentiate `∇J(θ)` we need to know about the environment dynamics but not always we do  
  -> **Policy Gradient Theorem**

### Policy Gradient Theorem
"For any differentiable policy and for any policy objective function, the policy gradient is: `∇J(θ) = E[∇ log π(a_t|s_t) R(τ)]`"

## The Reinforce Algorithm (Monte Carlo Reinforce)
Policy-gradient algorithm that uses an estimated return from an entire episode to update the policy parameter `θ`.
```
Training Loop:
  Use policy π to collect episode τ
  Use the episode to estimate the gradient "∇J(θ) ~= g = ∑ ∇logπ(a_t|s_t)R(τ)"
  Update the weights of the policy: "θ <- θ + α * g"
```

where in `g = ∑t ∇logπ(a_t|s_t) R(τ)`:  
- `∇logπ(a_t|s_t)`: the **direction of steepest increase** of the (log) probability of selecting action `a_t` from state `s_t`
- `R(τ)`:
  - If the return is high, it will push up the probabilities of the (state, action) combinations.
  - If the return is low, it will push down the probabilities of the (state, action) combinations.
 
**we can also collect multiple episodes (trajectories):**  
`g = (1/m) ∑i ∑t ∇logπ(a_ti|s_ti) R(τ_i)`  
`m`: number of trajectories


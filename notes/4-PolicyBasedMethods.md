# Policy Based Methods

The main idea is to parametrize the policy (using a neural network).  
The policy will output a **probability distribution over the actions**.  
`pi(s) = P[A|s]`  

## Policy-Based and Policy-Gradient Methods
The optimization is most of the time **on-policy**, in both cases we search directly for the optimal policy.  
The difference between these two methods lies on how we optimize the parameter `theta`:
- **Policy-Based Methods**: We optimize the parameter `theta` indirectly by maximizing the local approximation of the objective function.
- **Policy-Gradient Methods**: We optimize the parameter `theta` directly by performing the gradient ascent on the performance of the objective function `J(theta)`.

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

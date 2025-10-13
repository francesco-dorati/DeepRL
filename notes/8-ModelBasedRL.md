# Model Based Reinforcement Learning (MBRL)
**dynamics models**: `st+1 = f(st, at)`  

## The Process
- agent repeatedly tries to solve a problem -> **accumulates stata-action data**
- with that data the agent creates a **dynamics model**
- with this *dynamics model*  the agent **decides how to act by predicting the future**
- with those actions the agent collects more data and improves its model

## Academic definition
MBRL follows the framework of an agent **interacting with environment**, **learning a model** and **leveraging the model** for control decision.  
The agent acts in a **Markov Decision Process** governed by a transition function `st+1 = f(st, at)` and returns a reward at each step `r(st,at)`.  
With a collected dataset `D := Si, Ai, Si+1, Ri` the agent learns a model `St+1 = f(St,At)` **to minimize the negative log-likelyhood of the transition**.  

We employ sample-based model-predictive control (MCP) using a **learned dynamics model**, which **optimizes the expected reward** over a finite, recursively predicted horizon τ, from a set of actions sampled from a uniform distribution `U(a)`


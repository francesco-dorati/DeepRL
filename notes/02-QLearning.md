# Q-Learning
Off-policy value-based method that uses TD approach to train action-value function.
  
**Q-function**: action-value function that outputs the value of being in a state and taking an action from it.  
**Q-table**: table representation of the Q-function.

## Q-Learning Algorithm
```
INPUT: policy pi, num_episodes, alpha, epsilon
OUTPUT: value function Q

Initialize Q arbitrarily.
for i = 1 to num_episodes do:
  update epsilon
  repeat
    Choose action At
    Take action At and observe Rt+1, St+1
    Update Q(St, At)
  until St is terminal
```

### 1. Initialize Q
Initialize Q-table with random values or 0.  

### 2. Choose and Action
Choose an action using the policy derived from Q.  
Usually using **epsilon-greedy policy** and **decreasing epsilon over time**.

### 3. Perform Action and get Reward
Take action At and observe Rt+1, St+1.

### 4. Update Q(St, At)
**TDtarget**: `Rt+1 + gamma * max_a(Q(St+1, a))`  
**TDerror**: `TDtarget -  Q(St, At)`  
**Update**: `Q(St, At) <= Q(St, At) + alpha * TDerror`  

**Complete Formula**: `Q(St, At) <= Q(St, At) + alpha * [Rt+1 + gamma * max_a(Q(St+1, a)) - Q(St, At)]`

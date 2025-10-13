
# Deep Q-Learning

A neural network will approximate the Q function.  


### Deep Q-Network (DQN)
`state -> DQN -> [(action, q_value), (action, q_value), ...]`  

**ATARI example:**   
stacked frames -> CNN -> values  
**input preprocessing:** 160x210x3 -> 80x80x1 

### Deep Q-Learning
Create a loss function and use gradient descent to minimize it.  
**Q-Target (TD target):** `Rt+1 + gamma * argmax(Q(St+1, a))`  
**Q-Loss (TD target):** `Q-Target - Q(St, At)`  

**Two phases:**  
- **sampling:** perform actions and store the observed experience tuples in a replay memory.  
- **training:** select a small batch of tuples randomly and learn from this batch using a gradient descent update step.  
  
```
Initialize memory D to capacity N
Initialize DQN (Q) with random weights th
Initialize target DQN (Q') with weights th_m=th
for episode=1 to M:
  for t=1 to T:

    # SAMPLING
    Select action a_t with epsilon greedy policy
    Execute action a_t in emulator and observe reward r_t and image x_t+1
    Store transition (s_t, a_t, r_t, s_t+1) in D

    # TRAINING
    Sample random minibatch of transitions (s_j, a_j, r_j, s_j+1) from D
    Set y_j = r_j + gamma * argmax(Q'(s_t+1, a))
    Perform gradient descent step on (y_j - Q(s_j, a_j))^2 to Q's parameters

    Every C steps reset Q' = Q
  end
end
  ```

#### Experience Replay memory (D)
Since network tends to forget the previous experiences as it gets new experiences.  
Use a replay buffer that saves experience samples that can reused during training.  
=> Allows the agent to **learn from the same experiences multiple times**.

#### Fixed Q-Target (Q')
To calculate the loss (TD-Error), calculate the difference between TD target (Q-Target) and the current Q-value (estimation of Q).  
But if we use the same function for both tha Qtarget and the Qvalue it causes unstable training.   
We use a fixed network (Q') for estimating TD target and update it (copy Q parameters) every C steps.    
=> Stabilizes training

#### Double DQN
 If non-optimal actions are regularly given a higher Q value than the optimal best action, the learning will be complicated.  
- Use DQN network to **select the best action to take for the next state** (the action with the highest Q-value).  
- Use our Target network to **calculate the target Q-value of taking that action at the next state**.  
=> Helps reduce the overestimation of Q-values and train faster.





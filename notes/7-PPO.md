# Proximal Policy Optimization
Improves agent's training stability by **avoiding policy updates that are too large**.  
We use a ratio that indicates the difference between our current and old policy and clip this ratio to a specific range [1-ϵ, 1+ϵ]  
-> this way the Policy update is not too large and training is more stable.

## Clipped Surrogate Objective Function
This function will constrain the policy change in a small range using a clip.   
`L(θ) = E[ min( r(θ)At, clip(r(θ), 1-ϵ, 1+ϵ)At ) ]`  
where:  
- **ratio function**:
    `r(θ) = π(at|st) / πold(at|st)`
    Probability ratio between current and old policy.
    Way to estimate the divergence between old and current policy.
- **unclipped part**: `r(θ)At`
- **clipped part**: `clip(r(θ), 1-ϵ, 1+ϵ)At`
At the end we take the lower bound between clipped and unclipped parts  

We update the policy only if:  
- Ratio is in range `[1-ϵ, 1+ϵ]`  
- Ratio outside range but the advantage leads to getting closer to the range
  - Being below the ratio but the advantage is > 0
  - Being above the ratio but the advantage is < 0
-> **we restrict the range that the current policy can vary from the old one**

## Final Objective function
Combination of Clipped Surrogate Objective function, Value Loss Function and Entropy bonus   
`L(θ) = E[ Lclip(θ) - c1 * Lvf(θ) + c2 * S[π](st)`
- `Lvf(θ)`: **squared error value loss** (V(st) - Vtarg)^2
- `S[π](st)`: **entropy bonus** to ensure exploration

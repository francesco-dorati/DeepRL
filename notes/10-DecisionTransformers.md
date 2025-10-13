# Decision Transformers
Instead of training a policy using RL methods we use **sequence modeling algorithm (Transformer) that, given a desired return, past states, and actions, will generate future actions to achieve this desired return**.   
It is an autoregressive model conditioned on the desired return, past states, and actions to generate future actions that achieve desired return.

Uses generative trajectory modeling to replace conventional RL algorithms. **Decision Transformers** don't maximize return but generate a series of future actions that achieve desired reutrn.

[Decision Transformers Course](https://huggingface.co/blog/decision-transformers)

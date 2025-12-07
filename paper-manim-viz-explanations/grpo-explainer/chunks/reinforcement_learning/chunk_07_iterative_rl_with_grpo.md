# Iterative RL with GRPO

### Iterative RL with GRPO

As the reinforcement learning training process progresses, the old reward model may not be sufficient to supervise the current policy model.
Therefore, we also explore the iterative RL with GRPO.
As shown in Algorithm (ref: alg:iter-grpo), in iterative GRPO, we generate new training sets for the reward model based on the sampling results from the policy model and continually train the old reward model using a replay mechanism that incorporates 10% of historical data.
Then, we set the reference model as the policy model, and continually train the policy model with the new reward model.
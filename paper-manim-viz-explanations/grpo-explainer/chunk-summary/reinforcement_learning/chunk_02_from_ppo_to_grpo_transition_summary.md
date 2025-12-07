# From PPO to GRPO: Understanding Group-Relative Policy Optimization

## 1. Intuition & Core Idea

Imagine you're teaching a student to write better essays. With traditional reinforcement learning approaches like PPO (Proximal Policy Optimization), you might give feedback on each word they write - telling them whether using "fantastic" was better than "good" in a specific context. However, this approach has limitations when dealing with complex tasks involving multiple possible good solutions.

**The Problem**: Standard PPO treats all deviations from the old policy equally, regardless of whether they lead to better outcomes. It's like giving the same correction to a student who wrote a grammatically correct but boring essay versus one who took creative risks that paid off.

**The Solution - GRPO Insight**: Group-Relative Policy Optimization (GRPO) introduces a smarter approach by comparing policies within groups of similar trajectories. Instead of asking "Is this action better than what I did before?", GRPO asks "Among all the ways I could have responded to this situation, is this one of the better ones?"

Think of it like a writing coach who looks at 10 different student essays on the same topic and says "This particular essay structure works better than 8 out of 10 alternatives" rather than comparing it to just one previous attempt.

**Key Innovation**: GRPO uses **group-level importance sampling** - it considers how well a trajectory performs relative to other trajectories generated from the same prompt, making the learning process more robust and less dependent on having a perfect reference policy.

## 2. Technical Deep Dive

Let's break down the mathematical transition from PPO to GRPO:

### PPO Foundation
Traditional PPO optimizes the surrogate objective:

$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}{[q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ r_t(\theta) A_{t}, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_{t} \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})}$ is the probability ratio
- $A_t$ is the advantage at time step $t$
- $\epsilon$ is the clipping parameter (typically 0.2)

### Reward with KL Penalty
PPO typically uses rewards with KL regularization:

$$r_{t} = r_\phi(q, o_{\le t}) - \beta \log\frac{\pi_{\theta}(o_{t}|q, o_{<t})}{\pi_{ref}(o_{t}|q, o_{<t})}$$

Where:
- $r_\phi$ is the learned reward model
- $\pi_{ref}$ is the reference model (usually the supervised fine-tuned model)
- $\beta$ controls the strength of KL penalty

### GRPO Key Modifications

GRPO modifies PPO in several important ways:

1. **Group-Level Importance Sampling**: Instead of comparing to a single old policy, it compares within groups of trajectories
2. **Two-Sided Clipping**: Uses asymmetric clipping bounds $(1-\epsilon_{low}, 1+\epsilon_{high})$
3. **Flexible Loss Aggregation**: Supports different ways of aggregating losses across tokens

The core GRPO loss becomes:
$$\mathcal{L}_{GRPO} = -\mathbb{E}[\min(r_t^{group} A_t, \text{clip}(r_t^{group}, 1-\epsilon_{low}, 1+\epsilon_{high}) A_t)] + \beta D_{KL}(\pi_\theta || \pi_{ref})$$

## 3. Code Implementation Walkthrough

Let's examine how the theoretical concepts translate into code:

### Main Loss Computation (`_compute_loss` method)

```python
def _compute_loss(self, model, inputs):
    # Step 1: Prepare input data
    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    
    # Step 2: Get per-token log probabilities
    per_token_logps, entropies = self._get_per_token_logps_and_entropies(...)
    
    # Step 3: Compute KL divergence with reference model
    if self.beta != 0.0:
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        )
    
    # Step 4: Compute importance weights based on sampling level
    old_per_token_logps = inputs.get("old_per_token_logps", per_token_logps.detach())
    log_ratio = per_token_logps - old_per_token_logps
    
    if self.importance_sampling_level == "token":
        log_importance_weights = log_ratio  # Token-level comparison
    elif self.importance_sampling_level == "sequence":
        # Sequence-level: average across all tokens in sequence
        log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    
    # Step 5: Apply clipping and compute final loss
    coef_1 = torch.exp(log_importance_weights)
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    
    # Two-sided clipping with optional delta bound
    if self.args.delta is not None:
        coef_1 = torch.clamp(coef_1, max=self.args.delta)
    
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
```

### Key Implementation Details:

1. **Flexible Importance Sampling**: The code supports both token-level and sequence-level importance sampling through the `importance_sampling_level` parameter
2. **Asymmetric Clipping**: Uses separate parameters for lower (`epsilon_low`) and upper (`epsilon_high`) bounds
3. **KL Penalty Integration**: Properly computes and adds KL divergence penalty when `beta != 0.0`
4. **Multiple Loss Types**: Supports various aggregation strategies (GRPO, BNPO, DR-GRPO, DAPO)

### Reward Calculation (`_calculate_rewards` method)

```python
def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
    # Multiple reward functions can be combined
    rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
    
    # Process each reward function
    for i, (reward_func, ...) in enumerate(zip(self.reward_funcs, ...)):
        # Handle conversational vs regular text formats
        if is_conversational(inputs[0]):
            # Process as conversation
            messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
            texts = [apply_chat_template(...) for x in messages]
        else:
            # Process as plain text
            texts = [p + c for p, c in zip(prompts, completions)]
        
        # Compute rewards
        with torch.inference_mode():
            rewards_per_func[:, i] = reward_func(...).logits[:, 0]
```

## 4. Worked Example

Let's work through a concrete example with actual numbers:

### Setup
Consider a simple scenario:
- Batch size: 2 sequences
- Each sequence: 3 tokens
- Advantages: $A = [0.5, -0.3, 0.8]$ for sequence 1, $A = [0.2, 0.1, -0.4]$ for sequence 2
- Current policy log probs: $\log \pi_\theta = [-1.2, -0.8, -1.5]$ and $[-0.9, -1.1, -0.7]$
- Old policy log probs: $\log \pi_{old} = [-1.0, -1.0, -1.0]$ and $[-1.0, -1.0, -1.0]$
- Parameters: $\epsilon_{low} = 0.1$, $\epsilon_{high} = 0.2$, $\beta = 0.1$

### Step-by-Step Computation

**Step 1: Compute Log Ratios**
```python
log_ratios_seq1 = [-1.2 - (-1.0), -0.8 - (-1.0), -1.5 - (-1.0)] = [-0.2, 0.2, -0.5]
log_ratios_seq2 = [-0.9 - (-1.0), -1.1 - (-1.0), -0.7 - (-1.0)] = [0.1, -0.1, 0.3]
```

**Step 2: Compute Importance Weights (Token Level)**
```python
ratios_seq1 = [exp(-0.2), exp(0.2), exp(-0.5)] = [0.819, 1.221, 0.607]
ratios_seq2 = [exp(0.1), exp(-0.1), exp(0.3)] = [1.105, 0.905, 1.350]
```

**Step 3: Apply Clamping**
With $\epsilon_{low} = 0.1$, $\epsilon_{high} = 0.2$:
- Lower bound: $1 - 0.1 = 0.9$
- Upper bound: $1 + 0.2 = 1.2$

```python
clamped_seq1 = [clamp(0.819, 0.9, 1.2), clamp(1.221, 0.9, 1.2), clamp(0.607, 0.9, 1.2)]
               = [0.9, 1.2, 0.9]
clamped_seq2 = [clamp(1.105, 0.9, 1.2), clamp(0.905, 0.9, 1.2), clamp(1.350, 0.9, 1.2)]
               = [1.105, 0.905, 1.2]
```

**Step 4: Compute Loss Terms**
For sequence 1:
```python
loss1_terms = [0.9 * 0.5, 1.2 * (-0.3), 0.9 * 0.8] = [0.45, -0.36, 0.72]
loss2_terms = [0.9 * 0.5, 1.2 * (-0.3), 0.9 * 0.8] = [0.45, -0.36, 0.72]
final_loss_seq1 = [-min(0.45, 0.45), -min(-0.36, -0.36), -min(0.72, 0.72)] 
                 = [-0.45, 0.36, -0.72]
```

For sequence 2:
```python
loss1_terms = [1.105 * 0.2, 0.905 * 0.1, 1.2 * (-0.4)] = [0.221, 0.0905, -0.48]
loss2_terms = [1.105 * 0.2, 0.905 * 0.1, 1.2 * (-0.4)] = [0.221, 0.0905, -0.48]
final_loss_seq2 = [-min(0.221, 0.221), -min(0.0905, 0.0905), -min(-0.48, -0.48)]
                 = [-0.221, -0.0905, 0.48]
```

**Step 5: Average Loss**
```python
avg_loss_seq1 = (-0.45 + 0.36 - 0.72) / 3 = -0.27
avg_loss_seq2 = (-0.221 - 0.0905 + 0.48) / 3 = 0.0565
total_loss = (-0.27 + 0.0565) / 2 = -0.10675
```

(Note: Negative loss indicates improvement - we minimize the negative reward)

## 5. Mathematical Derivation

The transition from PPO to GRPO involves several key mathematical modifications:

### 1. Importance Sampling Ratio Modification

In PPO, the ratio is:
$$r_t^{PPO} = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

In GRPO, this becomes group-relative:
$$r_t^{GRPO} = \frac{\pi_\theta(a_t|s_t)}{\pi_{group}(a_t|s_t)}$$

Where $\pi_{group}$ represents the policy distribution over the group of similar trajectories.

### 2. Asymmetric Clipping Bounds

The standard PPO clipping:
$$\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$$

Becomes:
$$\text{clip}(r_t, 1-\epsilon_{low}, 1+\epsilon_{high})$$

This allows different sensitivity to positive vs negative deviations.

### 3. KL Divergence Penalty

The reverse KL divergence:
$$D_{KL}(\pi_{ref} || \pi_\theta) = \mathbb{E}_{a \sim \pi_\theta}[\log \frac{\pi_{ref}(a)}{\pi_\theta(a)}]$$

Is approximated using first-order Taylor expansion:
$$D_{KL} \approx \mathbb{E}[\exp(\log \pi_{ref} - \log \pi_\theta) - (\log \pi_{ref} - \log \pi_\theta) - 1]$$

## 6. Key Takeaways

### Most Important Points:
1. **Group-Relative Approach**: GRPO compares policies within groups of similar trajectories rather than to a fixed old policy
2. **Flexible Clipping**: Asymmetric bounds allow fine-tuned control over policy updates
3. **Multiple Aggregation Strategies**: Different loss types (GRPO, BNPO, etc.) offer trade-offs between stability and efficiency

### Common Pitfalls:
1. **Over-clipping**: Too aggressive clipping can prevent beneficial policy updates
2. **Reference Model Drift**: Ensure the reference model remains stable throughout training
3. **Reward Model Overfitting**: Monitor for reward hacking and implement proper regularization

### Practical Implementation Tips:
1. Start with conservative clipping bounds ($\epsilon_{low}=0.1$, $\epsilon_{high}=0.2$)
2. Tune $\beta$ carefully - too high causes conservatism, too low leads to instability
3. Monitor clipping statistics during training to adjust hyperparameters
4. Use sequence-level importance sampling for more stable gradients with long sequences

### Related Concepts:
- **PPO**: The foundational algorithm that GRPO extends
- **A3C/A2C**: Actor-critic methods that also use advantage estimation
- **Importance Sampling**: Statistical technique for efficient estimation
- **KL Regularization**: Common technique for preventing policy collapse

This transition from PPO to GRPO represents a sophisticated evolution in reinforcement learning for language models, providing better stability and performance in complex, multi-modal environments.
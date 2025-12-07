# Mixture of Experts: Experimental Results - Comprehensive Analysis

## 1. Intuition & Core Idea

The experimental results presented in this paper showcase the power of Reinforcement Learning (RL) for training mathematical reasoning models. Think of it like teaching a student math - instead of just showing them worked examples (supervised learning), we let them solve problems and then give them feedback on their solutions (reinforcement learning).

The key insight is that **reinforcement learning can significantly boost mathematical reasoning performance** even when training on the same dataset. The researchers took a base model (\spmath-Instruct 7B) and applied RL training (\spmath-RL 7B) using only Chain-of-Thought formatted data from GSM8K and MATH benchmarks. Despite having identical training data, the RL-enhanced model dramatically outperformed the supervised baseline.

This is remarkable because:
- **GSM8K**: 88.2% accuracy vs typical 7B models around 50-60%
- **MATH**: 51.7% accuracy vs typical 7B models around 10-20%

It's like taking a student who's learned from textbooks and then having them practice with a tutor who gives immediate feedback - the student improves dramatically even without new study materials.

## 2. Technical Deep Dive

The experimental success stems from the GRPO (Group Relative Policy Optimization) algorithm, which builds upon foundational RL principles:

### Mathematical Foundation

The core idea involves optimizing a policy $\pi_\theta$ to maximize expected rewards while staying close to a reference policy $\pi_{\text{ref}}$. The objective function is:

$$L(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} A^\pi(x,y) - \beta D_{\text{KL}}(\pi_{\text{ref}}(\cdot|x) \| \pi_\theta(\cdot|x)) \right]$$

Where:
- $x$: Input prompt (math problem)
- $y$: Generated completion (solution)
- $A^\pi(x,y)$: Advantage function measuring how much better action $y$ is than average
- $\beta$: KL penalty coefficient controlling policy deviation
- $D_{\text{KL}}$: Kullback-Leibler divergence

### GRPO Specific Innovations

GRPO introduces several enhancements:

1. **Group-based Advantage Calculation**: Instead of individual samples, advantages are computed within groups of multiple generations per prompt:
   $$A^{\text{grp}}(x,y) = r(x,y) - \frac{1}{N}\sum_{i=1}^{N} r(x,y_i)$$

2. **Clipped Objective**: To ensure stable training:
   $$L^{\text{GRPO}} = \min\left(\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} A^{\text{grp}}, \text{clip}\left(\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}, 1-\epsilon, 1+\epsilon\right) A^{\text{grp}}\right)$$

3. **Multi-objective Reward Combination**:
   $$R_{\text{total}} = \sum_{i=1}^{n} w_i R_i$$

These mathematical innovations allow the model to learn from relative performance within problem groups rather than absolute scores, leading to more stable and effective training.

## 3. Code Implementation Walkthrough

The implementation centers around the `GRPOTrainer` class, which orchestrates the RL training process:

### Core Components

1. **GRPOTrainer Class Initialization**:
```python
class GRPOTrainer(BaseTrainer):
    def __init__(
        self,
        model,           # Base language model
        reward_funcs,    # Reward functions for evaluation
        train_dataset,   # Training data with prompts
        args=None,       # GRPOConfig parameters
        ...
    ):
```

Key features:
- Supports both string model IDs and pre-loaded model objects
- Handles multiple reward functions (can combine different evaluation criteria)
- Configurable processing classes for different reward models

2. **GRPOConfig Class**:
This extensive configuration class controls every aspect of training:

```python
class GRPOConfig(TrainingArguments):
    # Model parameters
    beta = 0.0          # KL coefficient (0.0 disables reference model)
    
    # Data parameters  
    num_generations = 8 # Generations per prompt
    max_completion_length = 256
    
    # Training parameters
    epsilon = 0.2       # Clipping parameter
    loss_type = "dapo"  # Loss aggregation strategy
    
    # Generation parameters
    temperature = 1.0   # Sampling randomness
    use_vllm = False    # Accelerated generation
```

3. **Reward Functions**:
Customizable reward computation:
```python
def reward_func(completions, **kwargs):
    # Example: reward based on unique characters (dummy example)
    return [float(len(set(completion))) for completion in completions]
```

Real implementations would evaluate mathematical correctness, logical consistency, etc.

4. **Loss Computation**:
The `_compute_loss` method implements the mathematical GRPO objective:
```python
def _compute_loss(self, model, inputs):
    # 1. Generate completions for each prompt
    # 2. Compute rewards for all completions  
    # 3. Calculate group advantages
    # 4. Apply clipping and KL penalties
    # 5. Return final loss
```

## 4. Worked Example

Let's trace through a simplified example with concrete numbers:

### Setup
- **Prompt**: "Solve: 2x + 5 = 15"
- **Batch Size**: 2 prompts, 4 generations each
- **Reference Model**: Existing policy $\pi_{\text{ref}}$

### Step 1: Generation
```
Prompt 1: "Solve: 2x + 5 = 15"
Gen 1: "Subtract 5: 2x = 10. Divide by 2: x = 5"     (logprob: -2.1)
Gen 2: "2x = 10, so x = 5"                           (logprob: -2.8) 
Gen 3: "x = 5"                                       (logprob: -1.9)
Gen 4: "Wrong steps..."                              (logprob: -3.2)

Prompt 2: "Calculate 12 × 8"  
Gen 1: "12 × 8 = 96"                                 (logprob: -1.5)
Gen 2: "96"                                          (logprob: -1.2)
Gen 3: "100"                                         (logprob: -1.8) 
Gen 4: "95"                                          (logprob: -2.0)
```

### Step 2: Reward Calculation
Using a mathematical correctness reward function:
```
Rewards Prompt 1: [0.9, 0.8, 0.7, 0.1]  # Gen 4 incorrect
Rewards Prompt 2: [1.0, 0.9, 0.0, 0.0]  # Gen 3&4 wrong
```

### Step 3: Advantage Computation
Group advantages (reward - group mean):
```
Prompt 1 Mean: (0.9+0.8+0.7+0.1)/4 = 0.625
Advantages: [0.275, 0.175, 0.075, -0.525]

Prompt 2 Mean: (1.0+0.9+0.0+0.0)/4 = 0.475  
Advantages: [0.525, 0.425, -0.475, -0.475]
```

### Step 4: Policy Ratio Calculation
Importance sampling ratios:
```
Prompt 1 Ratios: [exp(-2.1+2.1), exp(-2.8+2.1), exp(-1.9+2.1), exp(-3.2+2.1)]
                = [1.00, 0.497, 1.22, 0.333]

Prompt 2 Ratios: [exp(-1.5+1.5), exp(-1.2+1.5), exp(-1.8+1.5), exp(-2.0+1.5)]  
                = [1.00, 1.35, 0.741, 0.607]
```

### Step 5: Clipped Loss Computation
With ε = 0.2, clipped ratios ∈ [0.8, 1.2]:
```
Sample 1: min(1.00×0.275, clip(1.00,0.8,1.2)×0.275) = 0.275
Sample 2: min(0.497×0.175, clip(0.497,0.8,1.2)×0.175) = 0.087
...
```

Final batch loss ≈ Average of all clipped surrogate objectives

This process iteratively pushes the model toward higher-reward completions while maintaining training stability through clipping.

## 5. Mathematical Derivation

### From Policy Gradient to GRPO

Starting with the policy gradient theorem:
$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(\tau) R(\tau)]$$

Importance sampling gives us:
$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_{\text{ref}}}[\frac{\pi_\theta(\tau)}{\pi_{\text{ref}}(\tau)} \nabla_\theta \log \pi_\theta(\tau) R(\tau)]$$

Using the log derivative trick:
$$= \mathbb{E}_{\tau \sim \pi_{\text{ref}}}[\frac{\pi_\theta(\tau)}{\pi_{\text{ref}}(\tau)} \nabla_\theta \log \pi_\theta(\tau) A^{\pi_{\text{ref}}}(\tau)]$$

GRPO modifies this by:
1. **Group Advantages**: Replace $A^{\pi_{\text{ref}}}(\tau)$ with group-relative advantages
2. **Clipping**: Apply PPO-style clipping to the ratio term
3. **Token-level**: Decompose at token level for fine-grained optimization

The final surrogate objective becomes:
$$L^{\text{GRPO}}(\theta) = \mathbb{E}\left[\sum_{t=1}^{T} \min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}$ and $\hat{A}_t$ are the estimated advantages.

## 6. Key Takeaways

### Critical Insights
1. **RL Effectiveness**: Reinforcement learning can dramatically improve performance even with identical training data
2. **Relative Feedback**: Group-based advantages provide more stable training signals than absolute rewards
3. **Scalability**: The approach works effectively at the 7B parameter scale, making it accessible

### Implementation Best Practices
- **Multiple Generations**: Need ≥2 generations per prompt for meaningful advantage calculation
- **Proper Clipping**: Essential for training stability (ε = 0.2 works well)
- **Reward Design**: Critical component - poor rewards lead to poor policies
- **KL Control**: Balance exploration vs. stability with appropriate β values

### Common Pitfalls
- Insufficient generations per prompt lead to noisy advantage estimates
- Poor reward function design can optimize for wrong objectives  
- Incorrect clipping can cause training instability
- Memory issues with reference model if not properly managed

### Further Reading
- **Foundational Papers**: PPO (Schulman et al., 2017), DPO (Rafailov et al., 2023)
- **Related Methods**: REINFORCE, A2C, TRPO
- **Applications**: Constitutional AI, Self-Improvement, Tool Usage

The experimental results demonstrate that careful RL implementation can unlock significant performance gains in specialized domains like mathematical reasoning, opening new pathways for developing capable AI systems.
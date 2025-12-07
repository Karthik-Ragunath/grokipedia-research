# Chain-of-Thought Reasoning Performance Analysis in Mixture of Experts Architectures

## 1. Intuition & Core Idea

Chain-of-Thought (CoT) reasoning is like teaching a student to solve complex math problems by showing their work step-by-step rather than just providing the final answer. Instead of asking "What is 12 × 15?" and expecting "180", we want the model to think: "First I'll multiply 12 × 10 = 120, then 12 × 5 = 60, so 120 + 60 = 180."

In mathematical reasoning, this step-by-step approach is crucial because:
- It helps identify where errors occur
- It demonstrates understanding of underlying concepts
- It makes complex problems more manageable
- It enables verification of the solution process

The research shows that \spmath-Instruct 7B achieves remarkable performance on competition-level math problems by leveraging advanced training techniques that optimize for this kind of sequential reasoning. Think of it as training a mathematician who doesn't just memorize formulas but actually understands how to break down complex problems systematically.

The key insight is that mathematical reasoning isn't just about getting the right answer—it's about developing the cognitive process of working through problems logically and sequentially.

## 2. Technical Deep Dive

### Mathematical Foundation of Reward-Based Training

The core mathematical framework involves optimizing a policy $\pi_\theta$ (our language model) to maximize expected rewards:

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}[r(x, y)]$$

Where:
- $x$: input prompt (mathematical problem)
- $y$: generated completion (step-by-step solution)
- $r(x, y)$: reward function evaluating solution correctness
- $\mathcal{D}$: distribution of training problems

### GRPO Loss Function

Group Relative Policy Optimization modifies the standard policy gradient approach:

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}\left[\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \cdot A(x, y)\right]$$

With clipping to ensure stability:

$$\mathcal{L}_{\text{GRPO}}^{\text{clipped}} = \mathbb{E}\left[\min\left(\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \cdot A(x, y), \text{clip}\left(\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}, 1-\epsilon, 1+\epsilon\right) \cdot A(x, y)\right)\right]$$

Where:
- $\pi_{\text{ref}}$: reference policy (typically previous version of model)
- $A(x, y)$: advantage function measuring how much better this completion is than average
- $\epsilon$: clipping parameter (default 0.2)

### Key Training Parameters

The configuration reveals several critical mathematical optimizations:

1. **Temperature (τ = 1.0)**: Controls randomness in generation
   $$P(y_i|y_{<i}) \propto \exp(\frac{\log P_{\text{model}}(y_i|y_{<i})}{\tau})$$

2. **Top-p sampling (p = 1.0)**: Nucleus sampling threshold for diversity

3. **KL Regularization (β = 0.0)**: Balances exploration vs reference policy adherence
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}} + \beta \cdot D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$$

## 3. Code Implementation Walkthrough

### GRPOTrainer Class Structure

```python
class GRPOTrainer(BaseTrainer):
    """
    Implements Group Relative Policy Optimization for mathematical reasoning
    """
```

The trainer inherits from `BaseTrainer` and specializes in reward-based optimization for mathematical tasks. Key components:

#### Reward Functions
```python
def reward_func(completions, **kwargs):
    # Custom logic to evaluate mathematical solutions
    return [float(len(set(completion))) for completion in completions]
```

This example shows a dummy reward function, but in practice this would evaluate:
- Correctness of mathematical steps
- Logical flow of reasoning
- Final answer accuracy

#### Configuration Parameters

The `GRPOConfig` class defines critical hyperparameters:

```python
# Generation parameters for diverse mathematical reasoning
temperature = 1.0        # Exploration vs exploitation balance
top_p = 1.0             # Nucleus sampling
max_completion_length = 256  # Allow sufficient space for CoT

# Training optimization parameters
beta = 0.0              # No KL regularization (memory efficiency)
num_iterations = 1      # Single iteration per batch
epsilon = 0.2           # Clipping bound for stability
```

### Key Implementation Features

1. **Multi-Reward Support**: Can combine multiple reward signals
2. **vLLM Integration**: Accelerated generation for large-scale training
3. **Importance Sampling**: Handles off-policy effects from generation frameworks
4. **Entropy Filtering**: Focuses training on high-uncertainty tokens

## 4. Worked Example

Let's walk through a concrete example of training on a mathematical problem:

### Problem: "Find the roots of x² - 5x + 6 = 0"

#### Step 1: Prompt Generation
```python
prompt = "Solve x² - 5x + 6 = 0 step by step."
```

#### Step 2: Model Generation (Multiple Samples)
```python
# Generated completions with different reasoning paths
completions = [
    "Step 1: Identify a=1, b=-5, c=6\nStep 2: Apply quadratic formula\nStep 3: x = (5±√(25-24))/2 = (5±1)/2\nStep 4: x = 3 or x = 2",
    "Step 1: Factor the equation\nStep 2: (x-2)(x-3) = 0\nStep 3: x = 2 or x = 3",
    "Step 1: Complete the square\nStep 2: (x-2.5)² = 0.25\nStep 3: x = 2.5 ± 0.5 = 2 or 3"
]
```

#### Step 3: Reward Calculation
```python
# Simplified reward function for demonstration
def math_reward_function(completions):
    rewards = []
    for completion in completions:
        score = 0
        
        # Check for correct final answer
        if "x = 2 or x = 3" in completion or "x = 3 or x = 2" in completion:
            score += 0.5
            
        # Check for logical steps
        if "quadratic formula" in completion.lower() or "factor" in completion.lower():
            score += 0.3
            
        # Check for mathematical correctness
        if "√(25-24)" in completion or "(x-2)(x-3)" in completion:
            score += 0.2
            
        rewards.append(score)
    
    return rewards

rewards = math_reward_function(completions)
# Result: [1.0, 1.0, 0.8] - All good solutions, third slightly less complete
```

#### Step 4: Policy Update Calculation

For the best completion (reward = 1.0):

```python
# Assume reference policy probability: 0.1
# Current policy probability: 0.3
# Advantage: 0.5 (better than average)

ratio = 0.3 / 0.1  # = 3.0
clipped_ratio = min(max(ratio, 1-0.2), 1+0.2)  # = 1.2

# Unclipped loss component: 3.0 * 0.5 = 1.5
# Clipped loss component: 1.2 * 0.5 = 0.6
# Final loss: min(1.5, 0.6) = 0.6
```

The clipping prevents overly aggressive updates when the current policy diverges significantly from the reference.

## 5. Mathematical Derivation

### Policy Gradient Theorem Application

Starting from the policy gradient theorem:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(\tau) \cdot R(\tau)\right]$$

For GRPO, we modify this with importance sampling relative to a reference policy:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\text{ref}}}\left[\frac{\pi_\theta(\tau)}{\pi_{\text{ref}}(\tau)} \nabla_\theta \log \pi_\theta(\tau) \cdot A(\tau)\right]$$

### Clipping Derivation

To ensure training stability, we clip the importance ratio:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r(\theta) \hat{A}, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \hat{A}\right)\right]$$

Where:
- $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}$ is the probability ratio
- $\hat{A}$ is the estimated advantage
- $\epsilon$ is the clipping parameter

This ensures that large policy updates are prevented, maintaining training stability while still allowing beneficial improvements.

## 6. Key Takeaways

### Main Insights

1. **Chain-of-Thought reasoning is fundamental**: Breaking down complex problems into logical steps dramatically improves mathematical performance, even outperforming much larger models without this capability.

2. **Reward-based training is essential**: The combination of diverse generation strategies and careful reward engineering enables models to learn sophisticated reasoning patterns.

3. **Configuration matters significantly**: Parameters like temperature, clipping bounds, and generation length directly impact the quality of learned reasoning processes.

### Common Pitfalls

1. **Overfitting to final answers**: Without proper reward design, models may learn to produce correct answers through memorization rather than genuine reasoning.

2. **Length bias**: Standard loss formulations can favor shorter completions with positive advantages, undermining thorough reasoning.

3. **Reference policy drift**: Without proper regularization, the policy can diverge too quickly from the reference, causing instability.

### Best Practices

1. **Use sequence-level rewards**: Evaluate entire reasoning chains rather than individual steps.

2. **Implement proper clipping**: Prevent catastrophic policy updates with carefully tuned clipping parameters.

3. **Balance exploration and exploitation**: Use appropriate temperature settings to maintain diversity in reasoning approaches.

### Further Reading

- [DeepSeekMath Paper](https://arxiv.org/abs/2402.03300): Original work on mathematical reasoning in language models
- [PPO Paper](https://arxiv.org/abs/1707.06347): Foundation for modern RLHF approaches
- [GRPO Paper](https://arxiv.org/abs/2405.14820): Specific algorithm implemented in the code
- [DAPO Paper](https://arxiv.org/abs/2503.14476): Advanced loss formulations for mathematical reasoning

The success of \spmath-Instruct demonstrates that with proper architectural choices and training methodology, relatively small models can achieve state-of-the-art mathematical reasoning capabilities through careful optimization of chain-of-thought processes.
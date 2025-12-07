# Shared Expert Isolation Context

It is worth noting that the prototype of shared expert isolation can be credited to [deepspeed_moe]. 
The key distinction lies in the fact that they derive this strategy from an engineering perspective, while we approach it from an algorithmic standpoint.

---

## Corresponding Code Implementation

**Relevance Score:** 8/10

**Explanation:** The paper chunk discusses "Shared Expert Isolation Context" which refers to the implementation of shared experts in MoE architectures. The code shows this through the DeepseekMoE class which explicitly handles both routed experts and shared experts, with the shared experts being processed separately and added to the final output. The configuration also supports shared experts through the n_shared_experts parameter.

### Code Section 1: The DeepseekMoE class implements the shared expert isolation context by maintaining separate processing paths for routed experts and shared experts. The shared experts are handled through the self.shared_experts attribute and added directly to the output (line 392-393), demonstrating the isolation concept where shared experts operate independently from the routed expert selection mechanism.

**File:** `modeling_deepseek.py` (lines 361-393)

```python
class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([DeepseekMLP(config, intermediate_size = config.moe_intermediate_size) for i in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(config=config, intermediate_size = intermediate_size)
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
```

### Code Section 2: The initialization of the DeepseekMoE class shows explicit handling of shared experts through the n_shared_experts configuration parameter. When configured, it creates a separate shared_experts MLP with an intermediate size scaled by the number of shared experts (lines 371-373), demonstrating the architectural separation of shared experts from routed experts.

**File:** `modeling_deepseek.py` (lines 365-374)

```python
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([DeepseekMLP(config, intermediate_size = config.moe_intermediate_size) for i in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(config=config, intermediate_size = intermediate_size)
    
```

### Code Section 3: The DeepseekConfig class defines the n_shared_experts parameter which controls the number of shared experts in the model. This configuration parameter enables the shared expert isolation functionality described in the paper chunk, allowing the model to be configured with different numbers of shared experts or disabled entirely (None) for dense models.

**File:** `configuration_deepseek.py` (lines 31-32)

```python
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts, None means dense model.
```

# Shared Expert Isolation Implementation

Towards this objective, in addition to the fine-grained expert segmentation strategy, we further isolate $K_{s}$ experts to serve as shared experts.
Regardless of the router module, each token will be deterministically assigned to these shared experts.
In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by $K_{s}$, as depicted in Figure (ref: fig:deepseek_moe)(c).

---

## Corresponding Code Implementation

**Relevance Score:** 8/10

**Explanation:** The code implements the shared expert isolation concept described in the paper chunk. The DeepseekMoE class handles shared experts by including them as a separate component that is always activated for all tokens, while maintaining constant computational cost by adjusting the number of routed experts.

### Code Section 1: The DeepseekMoE class implements the shared expert isolation mechanism. It initializes shared experts when configured (lines 371-373) and adds their output to the routed experts' output for all tokens (lines 391-392), demonstrating the deterministic assignment of tokens to shared experts.

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

### Code Section 2: The forward method shows how shared experts are integrated - they are applied to the identity input (line 392) regardless of routing decisions, while the routed experts follow the normal MoE pathway. This demonstrates the core concept of shared expert isolation.

**File:** `modeling_deepseek.py` (lines 375-393)

```python
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

### Code Section 3: The configuration parameter n_shared_experts controls the number of shared experts, which directly relates to the K_s experts mentioned in the paper chunk for shared expert isolation.

**File:** `configuration_deepseek.py` (lines 116-116)

```python
        n_shared_experts = None,
```

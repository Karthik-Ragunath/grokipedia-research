# Shared Expert Isolation Implementation

Towards this objective, in addition to the fine-grained expert segmentation strategy, we further isolate $K_{s}$ experts to serve as shared experts.
Regardless of the router module, each token will be deterministically assigned to these shared experts.
In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by $K_{s}$, as depicted in Figure (ref: fig:deepseek_moe)(c).
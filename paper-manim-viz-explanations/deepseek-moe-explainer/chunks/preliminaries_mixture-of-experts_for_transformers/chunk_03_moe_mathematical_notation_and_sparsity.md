# MoE Mathematical Notation and Sparsity

where $N$ denotes the total number of experts, 
$\operatorname{FFN}_{i}(\cdot)$ is the $i$-th expert FFN, 
$g_{i,t}$ denotes the gate value for the $i$-th expert, 
$s_{i,t}$ denotes the token-to-expert affinity, 
$\operatorname{Topk}(\cdot, K)$ denotes the set comprising $K$ highest affinity scores among those calculated for the $t$-th token and all $N$ experts,
and $\mathbf{e}_{i}^{l}$ is the centroid of the $i$-th expert in the $l$-th layer. 
Note that $g_{i,t}$ is sparse, indicating that only $K$ out of $N$ gate values are nonzero. 
This sparsity property ensures computational efficiency within an MoE layer, i.e., each token will be assigned to and computed in only $K$ experts.
Also, in the above formulations, we omit the layer normalization operation for brevity.
# Preliminaries: Mixture-of-Experts for Transformers

We first introduce a generic MoE architecture commonly used in Transformer language models. 
A standard Transformer language model is constructed by stacking $L$ layers of standard Transformer blocks, where each block can be represented as follows:

$$

    \mathbf{u}_{1:T}^{l} &= \operatorname{Self-Att}\left( \mathbf{h}_{1:T}^{l-1} \right) + \mathbf{h}_{1:T}^{l-1}, \\
    \mathbf{h}_{t}^{l} &= \operatorname{FFN}\left( \mathbf{u}_{t}^{l} \right) + \mathbf{u}_{t}^{l},

$$

where $T$ denotes the sequence length, 
$\operatorname{Self-Att}(\cdot)$ denotes the self-attention module, 
$\operatorname{FFN}(\cdot)$ denotes the Feed-Forward Network (FFN), 
$\mathbf{u}_{1:T}^{l} \in \mathbb{R}^{T \times d}$ are the hidden states of all tokens after the $l$-th attention module, 
and $\mathbf{h}_{t}^{l} \in \mathbb{R}^{d}$ is the output hidden state of the $t$-th token after the $l$-th Transformer block. 
For brevity, we omit the layer normalization in the above formulations. 

A typical practice to construct an MoE language model usually substitutes FFNs in a Transformer with MoE layers at specified intervals[switch,gshard,glam,st_moe].
An MoE layer is composed of multiple experts, where each expert is structurally identical to a standard FFN.
Then, each token will be assigned to one[switch] or two[gshard] experts. 
If the $l$-th FFN is substituted with an MoE layer, the computation for its output hidden state $\mathbf{h}_{t}^{l}$ is expressed as:

$$

\mathbf{h}_{t}^{l} & = \sum_{i=1}^{N} \left( {g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} \right) + \mathbf{u}_{t}^{l}, \\
g_{i,t} & = \begin{cases} 
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | 1 \leq j \leq N \}, K), \\
0, & \text{otherwise}, 
\end{cases} \\
s_{i,t} & = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l} \right), 

$$

where $N$ denotes the total number of experts, 
$\operatorname{FFN}_{i}(\cdot)$ is the $i$-th expert FFN, 
$g_{i,t}$ denotes the gate value for the $i$-th expert, 
$s_{i,t}$ denotes the token-to-expert affinity, 
$\operatorname{Topk}(\cdot, K)$ denotes the set comprising $K$ highest affinity scores among those calculated for the $t$-th token and all $N$ experts,
and $\mathbf{e}_{i}^{l}$ is the centroid of the $i$-th expert in the $l$-th layer. 
Note that $g_{i,t}$ is sparse, indicating that only $K$ out of $N$ gate values are nonzero. 
This sparsity property ensures computational efficiency within an MoE layer, i.e., each token will be assigned to and computed in only $K$ experts.
Also, in the above formulations, we omit the layer normalization operation for brevity. 

[Figure: 
Illustration of DeepSeekMoE. 
Subfigure (a) showcases an MoE layer with the conventional top-2 routing strategy. 
Subfigure (b) illustrates the fine-grained expert segmentation strategy. 
Subsequently, subfigure (c) demonstrates the integration of the shared expert isolation strategy, constituting the complete DeepSeekMoE architecture.
It is noteworthy that across these three architectures, the number of expert parameters and computational costs remain constant. 
]
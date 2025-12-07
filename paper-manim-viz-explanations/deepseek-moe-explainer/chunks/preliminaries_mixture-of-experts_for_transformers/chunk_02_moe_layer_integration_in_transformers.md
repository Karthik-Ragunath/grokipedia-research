# MoE Layer Integration in Transformers

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
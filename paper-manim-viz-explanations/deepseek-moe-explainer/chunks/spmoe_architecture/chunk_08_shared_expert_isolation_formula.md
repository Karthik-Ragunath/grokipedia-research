# Shared Expert Isolation Formula

With the shared expert isolation strategy integrated, an MoE layer in the complete DeepSeekMoE architecture is formulated as follows:

$$

\mathbf{h}_{t}^{l} & = \sum_{i=1}^{K_{s}} {\operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} + \sum_{i=K_{s} + 1}^{mN} \left( {g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} \right) + \mathbf{u}_{t}^{l}, \\
g_{i,t} & = \begin{cases} 
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | K_{s} + 1 \leq j \leq mN \}, mK - K_{s}), \\
0, & \text{otherwise}, 
\end{cases} \\
s_{i,t} & = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l} \right). 

$$

Finally, in DeepSeekMoE, the number of shared expert is $K_{s}$, 
the total number of routed experts is $mN - K_{s}$, 
and the number of nonzero gates is $mK - K_{s}$.
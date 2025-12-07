# Fine-Grained Expert Segmentation Formula

With the fine-grained expert segmentation, the output of an MoE layer can be expressed as:

$$

\mathbf{h}_{t}^{l} & = \sum_{i=1}^{mN} \left( {g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} \right) + \mathbf{u}_{t}^{l}, \\
g_{i,t} & = \begin{cases} 
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | 1 \leq j \leq mN \}, mK), \\
0, & \text{otherwise}, 
\end{cases} \\
s_{i,t} & = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l} \right), 

$$

where the total number of expert parameters is equal to $N$ times the number of parameters in a standard FFN, and $mN$ denotes the total number of fine-grained experts. 
With the fine-grained expert segmentation strategy, the number of nonzero gates will also increases to $mK$.
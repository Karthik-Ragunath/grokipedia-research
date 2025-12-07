# \spmoe{} Architecture

On top of the generic MoE architecture outlined in Section (ref: sec:preliminary), we introduce DeepSeekMoE, which is specifically designed to exploit the potential of expert specialization. 
As illustrated in Figure (ref: fig:deepseek_moe), our architecture incorporates two principal strategies: fine-grained expert segmentation and shared expert isolation. 
Both of these strategies are designed to elevate the level of expert specialization.

## Fine-Grained Expert Segmentation

In scenarios where the number of experts is limited, tokens assigned to a particular expert will be more likely to cover diverse types of knowledge. 
As a consequence, the designated expert will intend to learn vastly different types of knowledge in its parameters, and they are hard to be simultaneously utilized.
However, if each token can be routed to more experts, diverse knowledge will gain the potential to be decomposed and learned in different experts respectively. 
In this context, each expert can still retain a high level of expert specialization, contributing to a more focused knowledge distribution across experts.

In pursuit of the goal, while maintaining a consistent number of expert parameters and computational cost, we segment the experts with a finer grain. 
The finer expert segmentation enables a more flexible and adaptable combination of activated experts. 
To be specific, on top of a typical MoE architecture shown in Figure (ref: fig:deepseek_moe)(a), we segment each expert FFN into $m$ smaller experts by reducing the FFN intermediate hidden dimension to $\frac{1}{m}$ times its original size. 
Since each expert becomes smaller, in response, we also increase the number of activated experts to $m$ times to keep the same computation cost, as illustrated in Figure (ref: fig:deepseek_moe)(b).
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

From a combinatorial perspective, the fine-grained expert segmentation strategy substantially enhances the combinatorial flexibility of activated experts.
As an illustrative example, we consider the case where $N=16$. 
A typical top-2 routing strategy can yield $\binom{16}{2}=120$ possible combinations.
By contrast, if each expert is split into $4$ smaller experts, the fine-grained routing strategy can yield $\binom{64}{8}=4,426,165,368$ potential combinations. 
The surge in combinatorial flexibility enhances the potential for achieving more accurate and targeted knowledge acquisition. 

## Shared Expert Isolation

With a conventional routing strategy, tokens assigned to different experts may necessitate some common knowledge or information. 
As a result, multiple experts may converge in acquiring shared knowledge in their respective parameters, thereby resulting in redundancy in expert parameters. 
However, if there are shared experts dedicated to capturing and consolidating common knowledge across varying contexts, the parameter redundancy among other routed experts will be alleviated. 
This alleviation of redundancy will contribute to a more parameter-efficient model with more specialized experts.

Towards this objective, in addition to the fine-grained expert segmentation strategy, we further isolate $K_{s}$ experts to serve as shared experts.
Regardless of the router module, each token will be deterministically assigned to these shared experts.
In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by $K_{s}$, as depicted in Figure (ref: fig:deepseek_moe)(c).
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

It is worth noting that the prototype of shared expert isolation can be credited to [deepspeed_moe]. 
The key distinction lies in the fact that they derive this strategy from an engineering perspective, while we approach it from an algorithmic standpoint. 

## Load Balance Consideration

Automatically learned routing strategies may encounter the issue of load imbalance, which manifests two notable defects. 
Firstly, there is a risk of routing collapse[moe], i.e.,  the model always selects only a few experts, preventing other experts from sufficient training. 
Secondly, if experts are distributed across multiple devices, load imbalance can exacerbate computation bottlenecks.

**Expert-Level Balance Loss.**

In order to mitigate the risk of routing collapse, we also employ an expert-level balance loss. 
The computation of the balance loss is as follows:

$$

    \mathcal{L}_{\mathrm{ExpBal}} & = \alpha_1 \sum_{i=1}^{N^{\prime}}{f_i P_i}, \\
    f_i & = \frac{N^{\prime}}{K^{\prime}T} \sum_{t=1}^{T}{ \mathds{1}( \text{Token $t$ selects Expert $i$} )}, \\
    P_i & = \frac{1}{T} \sum_{t=1}^{T}{s_{i,t}},

$$

where $\alpha_1$ is a hyper-parameter called expert-level balance factor, 
$N^{\prime}$ is equal to $(mN - K_s)$ and $K^{\prime}$ is equal to $(mK - K_s)$ for brevity. 
$\mathds{1}(\cdot)$ denotes the indicator function. 

**Device-Level Balance Loss.**

In addition to the expert-level balance loss, we introduce a device-level balance loss. 
When aiming to alleviate computation bottlenecks, it becomes unnecessary to enforce strict balance constraints at the expert level, because excessive constraints on load balance will compromise model performance. 
Instead, our primary objective is to ensure balanced computation across the devices.
If we partition all routed experts into $D$ groups $\{\mathcal{E}_1, \mathcal{E}_2, ..., \mathcal{E}_D \}$, and deploy each group on a single device, the device-level balance loss is computed as follows:

$$

    \mathcal{L}_{\mathrm{DevBal}} & = \alpha_{2} \sum_{i=1}^{D}{f_i^{\prime} P_i^{\prime}}, \\
    f_i^{\prime} & = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i}{ f_j }, \\
    P_i^{\prime} & = \sum_{j \in \mathcal{E}_i}{ P_j },

$$

where $\alpha_{2}$ is a hyper-parameter called device-level balance factor. 
In practice, we set a small expert-level balance factor to mitigate the risk of routing collapse, and meanwhile set a larger device-level balance factor to promote balanced computation across the devices.
# Standard Transformer Architecture

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
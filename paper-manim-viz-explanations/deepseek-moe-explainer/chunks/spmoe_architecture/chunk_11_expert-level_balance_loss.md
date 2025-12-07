# Expert-Level Balance Loss

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
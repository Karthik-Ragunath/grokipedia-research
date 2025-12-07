# Device-Level Balance Loss

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
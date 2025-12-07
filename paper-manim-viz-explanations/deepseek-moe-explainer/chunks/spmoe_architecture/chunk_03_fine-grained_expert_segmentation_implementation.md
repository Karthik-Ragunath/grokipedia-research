# Fine-Grained Expert Segmentation Implementation

In pursuit of the goal, while maintaining a consistent number of expert parameters and computational cost, we segment the experts with a finer grain. 
The finer expert segmentation enables a more flexible and adaptable combination of activated experts. 
To be specific, on top of a typical MoE architecture shown in Figure (ref: fig:deepseek_moe)(a), we segment each expert FFN into $m$ smaller experts by reducing the FFN intermediate hidden dimension to $\frac{1}{m}$ times its original size. 
Since each expert becomes smaller, in response, we also increase the number of activated experts to $m$ times to keep the same computation cost, as illustrated in Figure (ref: fig:deepseek_moe)(b).
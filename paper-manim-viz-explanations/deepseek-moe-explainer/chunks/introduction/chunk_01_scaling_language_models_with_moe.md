# Scaling Language Models with MoE

Recent research and practices have empirically demonstrated that, with sufficient training data available, scaling language models with increased parameters and computational budgets can yield remarkably stronger models[gpt3,gpt4,llama,scaling_law]. 
It is imperative to acknowledge, however, that the endeavor to scale models to an extremely large scale is also associated with exceedingly high computational costs.
Considering the substantial costs, the Mixture-of-Experts (MoE) architecture[ori_moe1,ori_moe2,moe] has emerged as a popular solution.
It can enable parameter scaling, while concurrently keeping computational costs at a modest level.
Recent applications of MoE architectures in Transformers[transformer] have yielded successful attempts at scaling language models to a substantial size[switch,gshard,glam,st_moe], accompanied with remarkable performance. 
These achievements underscore the considerable potential and promise of MoE language models.
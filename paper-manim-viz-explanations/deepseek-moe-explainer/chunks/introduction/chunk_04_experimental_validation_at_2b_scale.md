# Experimental Validation at 2B Scale

Starting from a modest scale with 2B parameters, we validate the advantages of the DeepSeekMoE architecture. 
We conduct evaluations on 12 zero-shot or few-shot benchmarks spanning diverse tasks. 
Empirical results indicate that DeepSeekMoE 2B surpasses GShard 2B[gshard] by a substantial margin, and even matches GShard 2.9B, a larger MoE model with 1.5$\times$ expert parameters and computation. 
Remarkably, we find that DeepSeekMoE 2B nearly approaches the performance of its dense counterpart with an equivalent number of parameters, which sets the strict upper bound of MoE language models.
In pursuit of deeper insights, we conduct elaborate ablation studies and analysis on the expert specialization for DeepSeekMoE. 
These studies validate the effectiveness of fine-grained expert segmentation and shared expert isolation, and provide empirical evidence supporting the assertion that DeepSeekMoE can achieve a high level of expert specialization.
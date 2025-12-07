# Introduction - All Chunks

Total chunks: 6

---

## Chunk 1: Scaling Language Models with MoE

Recent research and practices have empirically demonstrated that, with sufficient training data available, scaling language models with increased parameters and computational budgets can yield remarkably stronger models[gpt3,gpt4,llama,scaling_law]. 
It is imperative to acknowledge, however, that the endeavor to scale models to an extremely large scale is also associated with exceedingly high computational costs.
Considering the substantial costs, the Mixture-of-Experts (MoE) architecture[ori_moe1,ori_moe2,moe] has emerged as a popular solution.
It can enable parameter scaling, while concurrently keeping computational costs at a modest level.
Recent applications of MoE architectures in Transformers[transformer] have yielded successful attempts at scaling language models to a substantial size[switch,gshard,glam,st_moe], accompanied with remarkable performance. 
These achievements underscore the considerable potential and promise of MoE language models.

---

## Chunk 2: Limitations of Existing MoE Architectures

Despite the promising potential of MoE architectures, existing MoE architectures potentially suffer from issues of knowledge hybridity and knowledge redundancy, which limit the expert specialization, i.e., each expert acquires non-overlapping and focused knowledge.
Conventional MoE architectures substitute the Feed-Forward Networks (FFNs) in a Transformer with MoE layers. 
Each MoE layer consists of multiple experts, with each structurally identical to a standard FFN, and each token is assigned to one[switch] or two[gshard] experts. 
This architecture manifests two potential issues:
(1) 
**Knowledge Hybridity**: existing MoE practices often employ a limited number of experts (e.g., 8 or 16), and thus tokens assigned to a specific expert will be likely to cover diverse knowledge. 
Consequently, the designated expert will intend to assemble vastly different types of knowledge in its parameters, which are hard to utilize simultaneously.
(2) 
**Knowledge Redundancy**: tokens assigned to different experts may require common knowledge. 
As a result, multiple experts may converge in acquiring shared knowledge in their respective parameters, thereby leading to redundancy in expert parameters. 
These issues collectively hinder the expert specialization in existing MoE practices, preventing them from reaching the theoretical upper-bound performance of MoE models.

---

## Chunk 3: DeepSeekMoE Architecture Introduction

In response to the aforementioned issues, we introduce **\spmoe{**}, an innovative MoE architecture specifically designed towards ultimate expert specialization. 
Our architecture involves two principal strategies: 
(1) **Fine-Grained Expert Segmentation:** 
while maintaining the number of parameters constant, we segment the experts into a finer grain by splitting the FFN intermediate hidden dimension. 
Correspondingly, keeping a constant computational cost, we also activate more fine-grained experts to enable a more flexible and adaptable combination of activated experts.
Fine-grained expert segmentation allows diverse knowledge to be decomposed more finely and be learned more precisely into different experts, where each expert will retain a higher level of specialization. 
In addition, the increased flexibility in combining activated experts also contributes to a more accurate and targeted knowledge acquisition.
(2) **Shared Expert Isolation:**
we isolate certain experts to serve as shared experts that are always activated, aiming at capturing and consolidating common knowledge across varying contexts. 
Through compressing common knowledge into these shared experts, redundancy among other routed experts will be mitigated. 
This can enhance the parameter efficiency and ensure that each routed expert retains specialized by focusing on distinctive aspects.
These architectural innovations in DeepSeekMoE offer opportunities to train a parameter-efficient MoE language model where each expert is highly specialized.

---

## Chunk 4: Experimental Validation at 2B Scale

Starting from a modest scale with 2B parameters, we validate the advantages of the DeepSeekMoE architecture. 
We conduct evaluations on 12 zero-shot or few-shot benchmarks spanning diverse tasks. 
Empirical results indicate that DeepSeekMoE 2B surpasses GShard 2B[gshard] by a substantial margin, and even matches GShard 2.9B, a larger MoE model with 1.5$\times$ expert parameters and computation. 
Remarkably, we find that DeepSeekMoE 2B nearly approaches the performance of its dense counterpart with an equivalent number of parameters, which sets the strict upper bound of MoE language models.
In pursuit of deeper insights, we conduct elaborate ablation studies and analysis on the expert specialization for DeepSeekMoE. 
These studies validate the effectiveness of fine-grained expert segmentation and shared expert isolation, and provide empirical evidence supporting the assertion that DeepSeekMoE can achieve a high level of expert specialization.

---

## Chunk 5: Scaling to 16B and Beyond

Leveraging our architecture, we subsequently scale up the model parameters to 16B and train DeepSeekMoE 16B on a large-scale corpus with 2T tokens. 
Evaluation results reveal that with only about 40% of computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B[deepseek_llm], a dense model trained on the same 2T corpus. 
We also compare DeepSeekMoE with open source models and the evaluations demonstrate that DeepSeekMoE 16B consistently outperforms models with a similar number of activated parameters by a large margin, and achieves comparable performance with LLaMA2 7B[llama2], which has approximately 2.5 times the activated parameters. 
Figure (ref: fig:openllm) demonstrates the evaluation results on the Open LLM Leaderboard (footnote: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 
Additionally, we conduct supervised fine-tuning (SFT) for alignment, transforming the model into a chat model.
Evaluation results show that DeepSeekMoE Chat 16B also achieves comparable performance with DeepSeek Chat 7B and LLaMA2 SFT 7B in the chat setting. 
Encouraged by these results, we further undertake a preliminary endeavor to scale up DeepSeekMoE to 145B. 
The experimental results still validate its substantial advantages over the GShard architecture consistently.
In addition, it shows performance comparable with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations.

---

## Chunk 6: Contributions Summary

Our contributions are summarized as follows: 

    
- **Architectural Innovation.**
    We introduce DeepSeekMoE, an innovative MoE architecture aiming at achieving ultimate expert specialization, which employs two principal strategies of fine-grained expert segmentation and shared expert isolation. 
    
- **Empirical Validation.**
    We conduct extensive experiments to empirically validate the effectiveness of the DeepSeekMoE architecture. 
    Experimental results validate the high level of expert specialization in DeepSeekMoE 2B, and indicate that DeepSeekMoE 2B can nearly approach the upper bound performance for MoE models
    
- **Scalability.**
    We scale up DeepSeekMoE to train a 16B model and show that with only about 40% of computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B and LLaMA2 7B. 
    We also undertake a preliminary endeavor to scale up DeepSeekMoE to 145B, highlighting its consistent advantages over the GShard architecture and showing a comparable performance with DeepSeek 67B.
    
- **Alignment for MoE.**
    We successfully perform supervised fine-tuning on DeepSeekMoE 16B to create an aligned chat model, showcasing the adaptability and versatility of DeepSeekMoE 16B.
    
- **Public Release.**
    In the spirit of open research, we release the model checkpoint of DeepSeekMoE 16B to the public. 
    Notably, this model can be deployed on a single GPU with 40GB of memory without the need for quantization.

---


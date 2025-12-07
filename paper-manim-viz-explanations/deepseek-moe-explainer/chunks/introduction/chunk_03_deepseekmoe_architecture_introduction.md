# DeepSeekMoE Architecture Introduction

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
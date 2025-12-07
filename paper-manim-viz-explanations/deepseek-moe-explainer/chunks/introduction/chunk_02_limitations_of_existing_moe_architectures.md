# Limitations of Existing MoE Architectures

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
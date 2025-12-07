# Combinatorial Flexibility Enhancement

From a combinatorial perspective, the fine-grained expert segmentation strategy substantially enhances the combinatorial flexibility of activated experts.
As an illustrative example, we consider the case where $N=16$. 
A typical top-2 routing strategy can yield $\binom{16}{2}=120$ possible combinations.
By contrast, if each expert is split into $4$ smaller experts, the fine-grained routing strategy can yield $\binom{64}{8}=4,426,165,368$ potential combinations. 
The surge in combinatorial flexibility enhances the potential for achieving more accurate and targeted knowledge acquisition.
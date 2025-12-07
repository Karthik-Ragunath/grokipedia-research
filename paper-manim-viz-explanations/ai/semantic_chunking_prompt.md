# Semantic Chunking Prompt

This prompt is used to split academic paper sections into semantically coherent chunks for further processing (e.g., visualization, summarization, or explanation generation).

## System Prompt

```
You are an expert at analyzing and decomposing academic papers into semantically coherent chunks. Your task is to split a section of a research paper into multiple smaller chunks, where each chunk represents a distinct concept, idea, or logical unit.

Guidelines for chunking:
1. Each chunk should be self-contained and represent ONE coherent concept or idea
2. Preserve the logical flow - chunks should follow the paper's narrative
3. Keep mathematical formulas together with their explanations
4. Group related definitions, theorems, or algorithms together
5. Separate high-level concepts from implementation details
6. Maintain context - include enough information for each chunk to be understandable
7. Target chunk sizes of 100-500 words, but prioritize semantic coherence over size
8. Preserve important technical terms and notation

Output Format:
Return the chunks in the following XML format:

<chunk id="1" title="[Brief descriptive title]">
[Content of the chunk]
</chunk>

<chunk id="2" title="[Brief descriptive title]">
[Content of the chunk]
</chunk>

... and so on

Important:
- The title should be a concise (3-8 words) description of the chunk's main topic
- Each chunk ID should be sequential (1, 2, 3, ...)
- Do NOT modify the content - preserve the original text, including any markdown or LaTeX notation
- Do NOT add explanations outside the chunk tags
```

## User Prompt Template

```
Please split the following section from an academic paper into semantically coherent chunks. The section is titled "{section_title}".

Section content:
---
{section_content}
---

Analyze the content and divide it into logical chunks where each chunk represents a distinct concept, idea, or logical unit. Ensure that mathematical formulas are kept with their explanations and that related concepts are grouped together.
```

## Example Usage

For a section about "Mixture of Experts", the output might look like:

```xml
<chunk id="1" title="Standard Transformer Architecture">
A standard Transformer language model is constructed by stacking L layers of standard Transformer blocks, where each block consists of a self-attention module followed by a Feed-Forward Network (FFN). The computation can be expressed as:
$$
\mathbf{u}_{1:T}^{l} = \operatorname{Self-Att}\left( \mathbf{h}_{1:T}^{l-1} \right) + \mathbf{h}_{1:T}^{l-1}
$$
$$
\mathbf{h}_{t}^{l} = \operatorname{FFN}\left( \mathbf{u}_{t}^{l} \right) + \mathbf{u}_{t}^{l}
$$
</chunk>

<chunk id="2" title="MoE Layer Substitution">
A typical practice to construct an MoE language model substitutes FFNs in a Transformer with MoE layers at specified intervals. An MoE layer is composed of multiple experts, where each expert is structurally identical to a standard FFN. Each token is then assigned to one or two experts based on a routing mechanism.
</chunk>

<chunk id="3" title="Token-to-Expert Routing">
The routing mechanism determines which experts process each token. The gate value for expert i is computed using a softmax over token-to-expert affinity scores:
$$
g_{i,t} = \begin{cases} 
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | 1 \leq j \leq N \}, K) \\
0, & \text{otherwise}
\end{cases}
$$
This sparsity ensures computational efficiency as each token is only processed by K out of N experts.
</chunk>
```

## Chunk Extraction Regex

To extract chunks from the LLM response, use this pattern:

```python
import re
pattern = r'<chunk\s+id="(\d+)"\s+title="([^"]+)">\s*(.*?)\s*</chunk>'
matches = re.findall(pattern, response, flags=re.DOTALL)
# matches will be: [(id, title, content), ...]
```


# Code-Chunk Aligner

Aligns paper chunks with corresponding code implementations using an LLM.

## Overview

This tool takes:
1. **Paper chunks** - Semantic chunks extracted from an academic paper
2. **Code files** - Implementation code (e.g., PyTorch model code)

And creates **aligned chunks** that include both the paper concept and the relevant code implementation.

## Usage

```bash
# Activate environment
pyenv activate deepseek-moe

# Navigate to aligner directory
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/deepseek-moe-explainer/aligner

# Run alignment
python code_chunk_aligner.py \
    --chunks-dir ../chunks \
    --code-dir ../deepseek-code \
    --output-dir ../chunks-with-code
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--chunks-dir, -c` | Directory containing paper chunks (required) |
| `--code-dir, -d` | Directory containing code files (required) |
| `--output-dir, -o` | Output directory for aligned chunks (required) |
| `--verbose, -v` | Print verbose output |

## Output Structure

```
chunks-with-code/
├── introduction/
│   ├── chunk_01_scaling_language_models_with_moe.md
│   └── ...
├── preliminaries.../
│   └── ...
└── spmoe_architecture/
    ├── chunk_01_deepseekmoe_overview.md
    ├── chunk_08_shared_expert_isolation_formula.md  # High relevance
    └── ...
```

## Output Format

Each aligned chunk contains:

1. **Original paper chunk content**
2. **Code Implementation section** (if relevant code found):
   - Relevance Score (0-10)
   - Explanation of how code relates to concept
   - Code sections with file paths and line numbers

Example output:

```markdown
# Shared Expert Isolation Formula

[Original paper content with formulas...]

---

## Corresponding Code Implementation

**Relevance Score:** 9/10

**Explanation:** The DeepseekMoE class implements the shared expert isolation...

### Code Section 1: Shared Experts in MoE Forward Pass

**File:** `modeling_deepseek.py` (lines 361-393)

```python
class DeepseekMoE(nn.Module):
    def __init__(self, config):
        ...
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekMLP(...)
    
    def forward(self, hidden_states):
        ...
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
```
```

## How It Works

1. **Reads code files** - Extracts all Python files and creates a summary of classes/functions
2. **Reads paper chunks** - Loads all semantic chunks from the chunks directory
3. **LLM Alignment** - For each chunk, queries NVIDIA Qwen Coder to find relevant code
4. **Creates output** - Generates new chunk files with aligned code sections

## LLM Configuration

- **Model**: `qwen/qwen3-coder-480b-a35b-instruct`
- **API**: NVIDIA Build API
- **Temperature**: 0.3 (low for consistent alignment)

## Requirements

- NVIDIA API key in `.env` file at project root
- Python packages: `openai`, `python-dotenv`

```bash
pip install openai python-dotenv
```


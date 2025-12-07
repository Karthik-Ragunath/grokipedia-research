# Paper Chunking Pipeline

A set of tools to extract sections from LaTeX papers and split them into semantically coherent chunks using NVIDIA's Qwen LLM.

## Overview

This pipeline consists of three main components:

1. **`latex_section_extractor.py`** - Extracts sections from LaTeX files and converts them to Markdown
2. **`semantic_chunker.py`** - Splits extracted text into semantic chunks using NVIDIA Qwen Coder API
3. **`paper_chunking_pipeline.py`** - Combined pipeline that runs both steps together

## Prerequisites

### 1. Activate Python Environment

```bash
pyenv activate deepseek-moe
```

### 2. Install Dependencies

```bash
pip install openai python-dotenv httpx
```

Or install from requirements.txt at project root:
```bash
pip install -r ../../requirements.txt
```

### 3. Set Up NVIDIA API Key

Create a `.env` file in the project root (`grokipedia-research/.env`) with:

```
NVIDIA_API_KEY=your_api_key_here
```

Get your API key from [NVIDIA Build](https://build.nvidia.com/).

## Usage

### Option 1: Full Pipeline (Recommended)

Run extraction and chunking in one command:

```bash
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/ai

python paper_chunking_pipeline.py \
    --latex-file ../deepseek-moe-explainer/deepseek-moe-paper/main.tex \
    --sections "Introduction,Preliminaries: Mixture-of-Experts for Transformers,DeepSeekMoE Architecture" \
    --output-dir ../deepseek-moe-explainer
```

**Options:**
- `--latex-file, -f` - Path to the LaTeX file
- `--sections, -s` - Comma-separated list of section names to extract
- `--output-dir, -o` - Base output directory for all generated files
- `--skip-chunking` - Only extract sections, skip LLM chunking step

### Option 2: Step-by-Step

#### Step 1: Extract Sections from LaTeX

```bash
python latex_section_extractor.py \
    --file ../deepseek-moe-explainer/deepseek-moe-paper/main.tex \
    --sections "Introduction,Preliminaries: Mixture-of-Experts for Transformers,DeepSeekMoE Architecture" \
    --output-dir ../deepseek-moe-explainer/extracted_sections
```

**Options:**
- `--file, -f` - Path to the LaTeX file
- `--sections, -s` - Comma-separated list of section names to extract
- `--output-dir, -o` - Output directory for extracted sections (default: `extracted_sections/` next to LaTeX file)

#### Step 2: Split into Semantic Chunks

**Single file:**
```bash
python semantic_chunker.py \
    --input ../deepseek-moe-explainer/extracted_sections/introduction.md \
    --output-dir ../deepseek-moe-explainer/chunks
```

**All files in a directory:**
```bash
python semantic_chunker.py \
    --input-dir ../deepseek-moe-explainer/extracted_sections \
    --output-dir ../deepseek-moe-explainer/chunks
```

**Options:**
- `--input, -i` - Path to a single markdown file to process
- `--input-dir, -d` - Path to directory containing markdown files to process
- `--output-dir, -o` - Output directory for chunks

## Output Structure

```
deepseek-moe-explainer/
├── extracted_sections/                    # Raw extracted sections as Markdown
│   ├── introduction.md
│   ├── preliminaries_mixture-of-experts_for_transformers.md
│   └── spmoe_architecture.md
│
└── chunks/                                # Semantic chunks organized by section
    ├── introduction/
    │   ├── _all_chunks.md                 # Combined view of all chunks
    │   ├── chunk_01_scaling_language_models_with_moe.md
    │   ├── chunk_02_limitations_of_existing_moe_architectures.md
    │   └── ...
    ├── preliminaries_mixture-of-experts_for_transformers/
    │   ├── _all_chunks.md
    │   └── ...
    └── spmoe_architecture/
        ├── _all_chunks.md
        └── ...
```

## Section Name Matching

The extractor uses fuzzy matching for section names, so these all work:
- `"Introduction"` → matches `\section{Introduction}`
- `"DeepSeekMoE Architecture"` → matches `\section{\spmoe{} Architecture}`
- `"Preliminaries"` → matches `\section{Preliminaries: Mixture-of-Experts for Transformers}`

## Prompt Template

The semantic chunking prompt is stored in `semantic_chunking_prompt.md`. You can modify this to customize how sections are split into chunks.

## Example: DeepSeek-MoE Paper

Complete example for the DeepSeek-MoE paper:

```bash
# Activate environment
pyenv activate deepseek-moe

# Navigate to ai directory
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/ai

# Run full pipeline
python paper_chunking_pipeline.py \
    --latex-file ../deepseek-moe-explainer/deepseek-moe-paper/main.tex \
    --sections "Introduction,Preliminaries: Mixture-of-Experts for Transformers,DeepSeekMoE Architecture" \
    --output-dir ../deepseek-moe-explainer

# View results
tree ../deepseek-moe-explainer/chunks
```

## Models Used

- **LLM**: `qwen/qwen3-coder-480b-a35b-instruct` via NVIDIA API
- **API Endpoint**: `https://integrate.api.nvidia.com/v1`

## Troubleshooting

### Missing NVIDIA API Key
```
Error: No NVIDIA_API_KEY found
```
→ Create `.env` file in project root with your API key

### Module Not Found
```
ModuleNotFoundError: No module named 'openai'
```
→ Run `pip install openai python-dotenv httpx`

### Section Not Found
```
Warning: Section 'XYZ' not found
```
→ Check exact section name in LaTeX file. The extractor handles `\spmoe{}` and similar commands.

---

## Code-Chunk Aligner

The `aligner/` subdirectory contains a script to align paper chunks with code implementations.

### Usage

```bash
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/ai/aligner

python code_chunk_aligner.py \
    --chunks-dir ../../deepseek-moe-explainer/chunks \
    --code-dir ../../deepseek-moe-explainer/deepseek-code \
    --output-dir ../../deepseek-moe-explainer/chunks-with-code
```

See `aligner/README.md` for detailed documentation.

---

## Manim Video Generator

The `manim_video_generator/` subdirectory contains a script to generate educational Manim videos from aligned chunks.

### Usage

```bash
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/ai/manim_video_generator

# Process all chunks
python chunk_video_generator.py \
    --chunks-dir ../../deepseek-moe-explainer/chunks-with-code \
    --output-dir ../../deepseek-moe-explainer/generated_videos

# Process a single chunk
python chunk_video_generator.py \
    --chunk-file ../../deepseek-moe-explainer/chunks-with-code/spmoe_architecture/chunk_06_shared_expert_isolation_motivation.md \
    --output-dir ../../deepseek-moe-explainer/generated_videos \
    --max-retries 5
```

See `manim_video_generator/README.md` for detailed documentation.

# Manim Video Generator

Generate educational Manim videos from aligned paper chunks (paper concepts + code implementations).

## Overview

This tool takes aligned chunks from the paper processing pipeline and generates animated educational videos explaining each concept, similar to 3Blue1Brown style.

## Prerequisites

### 1. Install Manim

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libcairo2-dev libpango1.0-dev ffmpeg

# Install manim
pip install manim
```

### 2. Activate Python Environment

```bash
pyenv activate deepseek-moe
```

### 3. Verify NVIDIA API Key

Ensure `.env` file exists at project root with `NVIDIA_API_KEY`.

## Usage

### Process All Chunks

```bash
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/ai/manim_video_generator

python chunk_video_generator.py \
    --chunks-dir ../../deepseek-moe-explainer/chunks-with-code \
    --output-dir ../../deepseek-moe-explainer/generated_videos
```

### Process Single Chunk

```bash
python chunk_video_generator.py \
    --chunk-file ../../deepseek-moe-explainer/chunks-with-code/spmoe_architecture/chunk_06_shared_expert_isolation_motivation.md \
    --output-dir ../../deepseek-moe-explainer/generated_videos
```

### With Options

```bash
python chunk_video_generator.py \
    --chunks-dir ../../deepseek-moe-explainer/chunks-with-code \
    --output-dir ../../deepseek-moe-explainer/generated_videos \
    --max-retries 5 \
    --quality m \
    --limit 3
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--chunks-dir, -d` | Directory containing aligned chunk files |
| `--chunk-file, -f` | Single chunk file to process |
| `--output-dir, -o` | Output directory for generated videos (required) |
| `--max-retries, -r` | Max retry attempts per chunk (default: 3) |
| `--quality, -q` | Video quality: `l`=480p, `m`=720p, `h`=1080p (default: l) |
| `--limit, -l` | Limit number of chunks to process |

## Output Structure

```
generated_videos/
├── manim_code/                     # Generated Python files
│   ├── chunk_01_xxx_20241207.py
│   ├── chunk_02_xxx_20241207.py
│   └── ...
└── videos/                         # Generated video files
    └── videos/
        └── chunk_01_xxx/
            └── 480p15/
                └── ConceptScene.mp4
```

## How It Works

1. **Reads aligned chunk** - Paper concept + relevant code implementation
2. **Generates prompt** - Creates a detailed prompt for Manim code generation
3. **Calls LLM** - Uses Qwen Coder 480B to generate Manim Python code
4. **Saves code** - Writes the code to a .py file
5. **Runs Manim** - Executes manim to render the animation
6. **Retries on failure** - If manim fails, regenerates code and retries

## Retry Mechanism

The generator includes automatic retry logic:

- If Manim execution fails (syntax error, runtime error, etc.)
- Regenerates new Manim code from scratch
- Tries up to `max_retries` times (default: 3)
- Logs each attempt and final success/failure

## Example Output

For a chunk about "Shared Expert Isolation", the generated video might show:

1. Title card with concept name
2. Visual representation of multiple experts
3. Animation showing shared knowledge being consolidated
4. Code snippet highlighting the implementation
5. Summary of the key idea

## Quality Settings

| Flag | Resolution | FPS | Use Case |
|------|------------|-----|----------|
| `l` | 480p | 15 | Fast preview, testing |
| `m` | 720p | 30 | Good quality, reasonable speed |
| `h` | 1080p | 60 | High quality, production |

## Troubleshooting

### "No Scene class found"
The generated code didn't include a proper Scene class. The code is regenerated on retry.

### Manim import errors
```bash
pip install manim
```

### ffmpeg not found
```bash
sudo apt install ffmpeg
```

### Video not generated
Check the `manim_code/` directory for the generated Python file and try running it manually:
```bash
manim -pql path/to/generated_code.py ConceptScene
```

## Model Used

- **LLM:** `qwen/qwen3-coder-480b-a35b-instruct`
- **API:** NVIDIA Build API


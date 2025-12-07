"""
Chunk Video Generator

This script takes aligned chunks (paper concepts + code) and generates
educational Manim videos explaining each chunk.

Usage:
    python chunk_video_generator.py \\
        --chunks-dir ../../deepseek-moe-explainer/chunks-with-code \\
        --output-dir ../../deepseek-moe-explainer/generated_videos

    # Process a single chunk
    python chunk_video_generator.py \\
        --chunk-file ../../deepseek-moe-explainer/chunks-with-code/spmoe_architecture/chunk_06_shared_expert_isolation_motivation.md \\
        --output-dir ../../deepseek-moe-explainer/generated_videos
"""

import asyncio
import argparse
import os
import re
import datetime
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load .env from project root (grokipedia-research/.env)
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# Debug: Print if API key is found
if not os.getenv("NVIDIA_API_KEY"):
    print(f"WARNING: NVIDIA_API_KEY not found. Checked: {env_path}")
    print("Make sure .env file exists at the project root with NVIDIA_API_KEY=your_key")


# System prompt for Manim code generation
MANIM_SYSTEM_PROMPT = """You are an expert in creating educational animations using Manim (Mathematical Animation Engine), similar to 3Blue1Brown style. You specialize in:
- Breaking down complex ML/AI concepts into visual steps with clear narratives
- Using Text() for all regular text (faster, no LaTeX compilation needed)
- Using MathTex() ONLY for mathematical formulas (e.g., equations, symbols)
- Using appropriate Manim primitives (Circle, Rectangle, Arrow, VGroup, NumberLine, Axes)
- Creating smooth, educational animations with proper timing and wait periods
- CLEANING UP content between steps using FadeOut() to prevent visual clutter
- Grouping related objects in VGroup() for easier animation and removal
- Writing clean, executable Manim code following best practices
- Using the Scene class with proper construct() methods
- Implementing effective visual metaphors and transitions
- Using colors to highlight important concepts
- Following Manim Community Edition (manim-ce) syntax

Generate production-ready code that can be directly executed with: manim -pql scene.py SceneName

CRITICAL RULES:
1. Use Text() for words and sentences. Use MathTex() ONLY for mathematical notation.
2. Always fade out or remove previous content before introducing new content in each step.
3. Keep the scene clean - use self.play(FadeOut(objects)) between major transitions.
4. AVOID Code() objects - they have complex parameters that often fail. Use Text() with monospace styling instead.
5. For showing code snippets, use Text() with font="Monospace" and smaller font size.
6. Test your knowledge - only use Manim Community Edition v0.19.0 compatible methods and parameters.
7. Keep text short and readable - break long sentences into multiple lines.
8. Use scale() to make sure text fits on screen.
9. The class name MUST be exactly "ConceptScene" - do not use any other name."""


def create_manim_prompt(chunk_content: str, chunk_title: str) -> str:
    """Create a prompt for generating Manim code from an aligned chunk."""
    
    return f"""Generate complete, executable Manim Python code to create an educational animation explaining this concept from an ML research paper.

## Concept Title: {chunk_title}

## Content (includes paper explanation and relevant code):
{chunk_content}

## Requirements:
- Create a Scene class named EXACTLY "ConceptScene" that inherits from Scene
- Use Text() for regular text (NOT Tex()). Text() is faster and doesn't require LaTeX
- ONLY use MathTex() for mathematical formulas and equations
- Break down the concept into 3-5 clear visual steps
- Show the key idea/concept first, then illustrate with diagrams
- If code is included, show simplified code snippets using Text() with font="Monospace"
- Use appropriate Manim shapes (Circle, Rectangle, Square, Arrow, Line, Dot, etc.)
- Include smooth animations (Write, FadeIn, FadeOut, Transform, Create, GrowArrow, etc.)
- Use colors effectively (BLUE, RED, GREEN, YELLOW, ORANGE, PURPLE, etc.)
- Position elements using .to_edge(), .shift(), .next_to(), .move_to()
- Include self.wait() between major steps (0.5-1 second)
- CRITICAL: Use FadeOut() or self.remove() to clean up previous content before showing new content
- Group related objects in VGroup() so they can be animated together
- Keep the scene clean - don't let objects accumulate and clutter the frame
- Make it educational and engaging with clear visual metaphors
- Ensure the code is syntactically correct and ready to run
- The animation should be 15-45 seconds long
- Import only what you need: from manim import *

IMPORTANT: The class MUST be named "ConceptScene" - this is required for the automation.

Return ONLY the Python code, no explanations before or after the code block."""


async def generate_manim_code(prompt: str, system_prompt: str = MANIM_SYSTEM_PROMPT) -> str:
    """Generate Manim code using the Qwen coder model."""
    
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    print("  Generating Manim code with Qwen Coder...")
    
    completion = client.chat.completions.create(
        model="qwen/qwen3-coder-480b-a35b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=4096,
        stream=True
    )
    
    full_content = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_content += content
    
    print("\n")
    return full_content


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks or raw text."""
    # Try to extract from markdown code blocks first
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[0].strip()
    # If no code blocks found, return the raw text
    return text.strip()


def save_manim_code(code: str, chunk_output_dir: Path) -> str:
    """Save Manim code to a Python file in the chunk's output directory."""
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = chunk_output_dir / "manim_code.py"
    
    # Extract clean Python code
    clean_code = extract_python_code(code)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(clean_code)
    
    print(f"  Manim code saved to: {filepath}")
    return str(filepath)


def run_manim_scene(scene_filepath: str, chunk_output_dir: Path, quality: str = "l") -> Dict:
    """
    Run manim to generate video from the scene file.
    
    Args:
        scene_filepath: Path to the Python file containing the Scene
        chunk_output_dir: Directory for this chunk's output
        quality: Video quality - 'l' (low/480p), 'm' (medium/720p), 'h' (high/1080p)
    
    Returns:
        Dict with paths to generated video and assets
    """
    # Use a temp media dir for manim, then copy results
    media_dir = chunk_output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the file to find scene class name
    with open(scene_filepath, 'r') as f:
        content = f.read()
    
    # Look for class definitions that inherit from Scene
    scene_match = re.search(r'class\s+(\w+)\s*\(\s*(?:Scene|ThreeDScene|MovingCameraScene)\s*\)', content)
    if not scene_match:
        raise ValueError("No Scene class found in the generated code")
    scene_name = scene_match.group(1)
    
    print(f"  Running manim for scene: {scene_name}")
    
    # Quality settings
    quality_flags = {
        'l': '-pql',   # 480p15
        'm': '-pqm',   # 720p30
        'h': '-pqh',   # 1080p60
    }
    quality_dirs = {
        'l': '480p15',
        'm': '720p30',
        'h': '1080p60',
    }
    
    # Run manim command using python -m manim (more reliable than just 'manim')
    cmd = [
        "python", "-m", "manim",
        quality_flags.get(quality, '-pql'),
        "--media_dir", str(media_dir),
        scene_filepath,
        scene_name
    ]
    
    print(f"  Running command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
    
    if result.stdout:
        print(result.stdout)
    
    # Find the generated video file
    scene_file_basename = os.path.splitext(os.path.basename(scene_filepath))[0]
    quality_dir = quality_dirs.get(quality, '480p15')
    
    # Look for video in manim's output structure
    source_video = None
    for qd in quality_dirs.values():
        potential_path = media_dir / "videos" / scene_file_basename / qd / f"{scene_name}.mp4"
        if potential_path.exists():
            source_video = potential_path
            break
    
    if not source_video:
        raise FileNotFoundError(f"Video not found in {media_dir}")
    
    # Copy video to chunk directory as video.mp4
    import shutil
    dest_video = chunk_output_dir / "video.mp4"
    shutil.copy2(source_video, dest_video)
    print(f"  Video copied to: {dest_video}")
    
    # Copy text assets (SVG files) if they exist
    texts_dir = media_dir / "texts"
    if texts_dir.exists():
        dest_texts = chunk_output_dir / "texts"
        if dest_texts.exists():
            shutil.rmtree(dest_texts)
        shutil.copytree(texts_dir, dest_texts)
        print(f"  Text assets copied to: {dest_texts}")
    
    # Copy image assets if they exist
    images_dir = media_dir / "images"
    if images_dir.exists():
        dest_images = chunk_output_dir / "images"
        if dest_images.exists():
            shutil.rmtree(dest_images)
        shutil.copytree(images_dir, dest_images)
        print(f"  Image assets copied to: {dest_images}")
    
    # Clean up the media directory to save space (keep only essential files)
    # Remove partial movie files
    partial_dir = media_dir / "videos" / scene_file_basename / quality_dir / "partial_movie_files"
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    
    return {
        'video': str(dest_video),
        'texts': str(chunk_output_dir / "texts") if texts_dir.exists() else None,
        'images': str(chunk_output_dir / "images") if images_dir.exists() else None
    }


def read_chunk_file(filepath: str) -> tuple[str, str]:
    """Read a chunk file and extract title and content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to extract title from first heading
    title_match = re.match(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
    else:
        # Use filename as title
        title = Path(filepath).stem.replace('_', ' ').title()
    
    return title, content


async def process_chunk(
    chunk_file: str,
    output_dir: Path,
    section_name: str = None,
    max_retries: int = 3,
    quality: str = "l"
) -> Dict:
    """
    Process a single chunk and generate video.
    
    Args:
        chunk_file: Path to the chunk markdown file
        output_dir: Base directory for output
        section_name: Name of the section (e.g., 'introduction', 'spmoe_architecture')
        max_retries: Number of retry attempts if generation fails
        quality: Video quality setting
    
    Returns:
        Dictionary with results
    """
    chunk_title, chunk_content = read_chunk_file(chunk_file)
    chunk_name = Path(chunk_file).stem
    
    # Create output directory structure matching chunks-with-code
    # e.g., generated_videos/spmoe_architecture/chunk_01_xxx/
    if section_name:
        chunk_output_dir = output_dir / section_name / chunk_name
    else:
        chunk_output_dir = output_dir / chunk_name
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {chunk_title}")
    print(f"Output dir: {chunk_output_dir}")
    print(f"{'='*60}")
    
    last_error = None
    scene_filepath = None
    video_result = None
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n  Attempt {attempt}/{max_retries}")
            
            # Generate Manim code
            prompt = create_manim_prompt(chunk_content, chunk_title)
            manim_code = await generate_manim_code(prompt)
            
            # Save the code to a file in the chunk's directory
            scene_filepath = save_manim_code(manim_code, chunk_output_dir)
            
            # Run manim to generate video
            video_result = run_manim_scene(scene_filepath, chunk_output_dir, quality)
            
            # Success!
            print(f"\n  ✓ SUCCESS! Video generated on attempt {attempt}/{max_retries}")
            
            return {
                'chunk_file': chunk_file,
                'chunk_title': chunk_title,
                'chunk_output_dir': str(chunk_output_dir),
                'scene_file': scene_filepath,
                'video_path': video_result['video'],
                'texts_path': video_result.get('texts'),
                'images_path': video_result.get('images'),
                'success': True,
                'attempts': attempt
            }
            
        except subprocess.CalledProcessError as e:
            last_error = e
            print(f"\n  ✗ FAILED: Manim execution error")
            print(f"    Error: {e.stderr[:500] if e.stderr else str(e)}")
            
            if attempt < max_retries:
                print(f"    Retrying... ({max_retries - attempt} attempts remaining)")
            
        except Exception as e:
            last_error = e
            print(f"\n  ✗ FAILED: {type(e).__name__}")
            print(f"    Error: {str(e)[:500]}")
            
            if attempt < max_retries:
                print(f"    Retrying... ({max_retries - attempt} attempts remaining)")
    
    # All retries exhausted
    print(f"\n  ✗ Max retries ({max_retries}) reached. Giving up on this chunk.")
    
    return {
        'chunk_file': chunk_file,
        'chunk_title': chunk_title,
        'chunk_output_dir': str(chunk_output_dir),
        'scene_file': scene_filepath,
        'video_path': None,
        'success': False,
        'attempts': max_retries,
        'error': str(last_error)
    }


async def process_chunks_directory(
    chunks_dir: str,
    output_dir: str,
    max_retries: int = 3,
    quality: str = "l",
    limit: int = None
) -> List[Dict]:
    """
    Process all chunks in a directory, organizing output by section.
    
    Args:
        chunks_dir: Directory containing chunk markdown files (organized by section)
        output_dir: Directory for output (will mirror section structure)
        max_retries: Number of retry attempts per chunk
        quality: Video quality setting
        limit: Maximum number of chunks to process (None = all)
    
    Returns:
        List of result dictionaries
    """
    chunks_path = Path(chunks_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("MANIM VIDEO GENERATOR")
    print(f"{'='*70}")
    print(f"\nChunks directory: {chunks_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max retries: {max_retries}")
    print(f"Quality: {quality}")
    
    # Find all chunk files with their section names
    chunk_entries = []  # List of (section_name, chunk_file_path)
    for section_dir in sorted(chunks_path.iterdir()):
        if section_dir.is_dir():
            section_name = section_dir.name
            for chunk_file in sorted(section_dir.glob('chunk_*.md')):
                chunk_entries.append((section_name, str(chunk_file)))
    
    if limit:
        chunk_entries = chunk_entries[:limit]
    
    print(f"\nFound {len(chunk_entries)} chunk files to process")
    
    # Show sections
    sections = set(entry[0] for entry in chunk_entries)
    print(f"Sections: {', '.join(sorted(sections))}")
    
    # Process each chunk
    results = []
    for i, (section_name, chunk_file) in enumerate(chunk_entries, 1):
        print(f"\n[{i}/{len(chunk_entries)}]", end="")
        result = await process_chunk(
            chunk_file=chunk_file,
            output_dir=output_path,
            section_name=section_name,
            max_retries=max_retries,
            quality=quality
        )
        results.append(result)
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nTotal: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successfully generated videos:")
        for r in successful:
            print(f"  - {r['chunk_title']}")
            print(f"    Output: {r['chunk_output_dir']}")
    
    if failed:
        print(f"\n✗ Failed chunks:")
        for r in failed:
            print(f"  - {r['chunk_title']}")
            print(f"    Error: {r.get('error', 'Unknown')[:100]}")
    
    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Generate Manim educational videos from aligned paper chunks"
    )
    parser.add_argument(
        "--chunks-dir", "-d",
        type=str,
        default=None,
        help="Directory containing aligned chunk files"
    )
    parser.add_argument(
        "--chunk-file", "-f",
        type=str,
        default=None,
        help="Single chunk file to process"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for generated videos"
    )
    parser.add_argument(
        "--max-retries", "-r",
        type=int,
        default=3,
        help="Maximum retry attempts per chunk (default: 3)"
    )
    parser.add_argument(
        "--quality", "-q",
        type=str,
        choices=['l', 'm', 'h'],
        default='l',
        help="Video quality: l=480p, m=720p, h=1080p (default: l)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of chunks to process"
    )
    
    args = parser.parse_args()
    
    if not args.chunks_dir and not args.chunk_file:
        parser.error("Either --chunks-dir or --chunk-file must be specified")
    
    if args.chunk_file:
        # Process single chunk - extract section name from path
        chunk_path = Path(args.chunk_file)
        section_name = chunk_path.parent.name  # e.g., 'spmoe_architecture'
        
        result = await process_chunk(
            chunk_file=args.chunk_file,
            output_dir=Path(args.output_dir),
            section_name=section_name,
            max_retries=args.max_retries,
            quality=args.quality
        )
        
        if result['success']:
            print(f"\n✓ Video generated!")
            print(f"  Output directory: {result['chunk_output_dir']}")
            print(f"  Video: {result['video_path']}")
        else:
            print(f"\n✗ Failed to generate video: {result.get('error', 'Unknown')}")
    
    else:
        # Process directory
        await process_chunks_directory(
            chunks_dir=args.chunks_dir,
            output_dir=args.output_dir,
            max_retries=args.max_retries,
            quality=args.quality,
            limit=args.limit
        )


if __name__ == "__main__":
    asyncio.run(main())


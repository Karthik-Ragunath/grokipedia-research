"""
Semantic Chunker

This script takes extracted text sections and uses NVIDIA's Qwen model
to split them into semantically coherent chunks.

Usage:
    python semantic_chunker.py --input extracted_section.md --output-dir ./chunks
    python semantic_chunker.py --input-dir ./extracted_sections --output-dir ./chunks
"""

import asyncio
import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI
import httpx
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


# System prompt for semantic chunking
SYSTEM_PROMPT = """You are an expert at analyzing and decomposing academic papers into semantically coherent chunks. Your task is to split a section of a research paper into multiple smaller chunks, where each chunk represents a distinct concept, idea, or logical unit.

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
- Do NOT add explanations outside the chunk tags"""


def create_user_prompt(section_title: str, section_content: str) -> str:
    """Create the user prompt for semantic chunking."""
    return f"""Please split the following section from an academic paper into semantically coherent chunks. The section is titled "{section_title}".

Section content:
---
{section_content}
---

Analyze the content and divide it into logical chunks where each chunk represents a distinct concept, idea, or logical unit. Ensure that mathematical formulas are kept with their explanations and that related concepts are grouped together."""


async def generate_chunks(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Generate semantic chunks using NVIDIA's Qwen Coder model."""
    
    # Create client for NVIDIA API
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    print("Calling NVIDIA Qwen Coder API for semantic chunking...")
    
    completion = client.chat.completions.create(
        model="qwen/qwen3-coder-480b-a35b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for more consistent chunking
        top_p=0.8,
        max_tokens=8192,  # Allow longer responses for full chunking
        stream=True
    )
    
    full_content = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_content += content
    
    print("\n")  # New line after streaming
    return full_content


def extract_chunks_from_response(response: str) -> List[Dict]:
    """Extract chunks from the LLM response using regex."""
    pattern = r'<chunk\s+id="(\d+)"\s+title="([^"]+)">\s*(.*?)\s*</chunk>'
    matches = re.findall(pattern, response, flags=re.DOTALL)
    
    chunks = []
    for match in matches:
        chunk_id, title, content = match
        chunks.append({
            'id': int(chunk_id),
            'title': title.strip(),
            'content': content.strip()
        })
    
    # Sort by ID to ensure order
    chunks.sort(key=lambda x: x['id'])
    
    return chunks


def read_markdown_file(filepath: str) -> tuple[str, str]:
    """Read a markdown file and extract title and content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to extract title from first heading
    title_match = re.match(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
        # Remove the title line from content
        content = re.sub(r'^#\s+.+\n*', '', content, count=1)
    else:
        # Use filename as title
        title = Path(filepath).stem.replace('_', ' ').title()
    
    return title, content.strip()


def save_chunks(chunks: List[Dict], output_dir: Path, section_name: str) -> List[str]:
    """Save chunks to individual markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Create a subdirectory for this section's chunks
    section_safe_name = re.sub(r'[^\w\s-]', '', section_name)
    section_safe_name = re.sub(r'\s+', '_', section_safe_name).lower()
    section_dir = output_dir / section_safe_name
    section_dir.mkdir(parents=True, exist_ok=True)
    
    for chunk in chunks:
        # Create filename from chunk ID and title
        title_safe = re.sub(r'[^\w\s-]', '', chunk['title'])
        title_safe = re.sub(r'\s+', '_', title_safe).lower()
        filename = f"chunk_{chunk['id']:02d}_{title_safe}.md"
        filepath = section_dir / filename
        
        # Write chunk content
        chunk_content = f"# {chunk['title']}\n\n{chunk['content']}"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(chunk_content)
        
        saved_files.append(str(filepath))
        print(f"Saved chunk {chunk['id']}: {filepath}")
    
    # Also save a combined file with all chunks
    combined_filepath = section_dir / "_all_chunks.md"
    with open(combined_filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {section_name} - All Chunks\n\n")
        f.write(f"Total chunks: {len(chunks)}\n\n---\n\n")
        for chunk in chunks:
            f.write(f"## Chunk {chunk['id']}: {chunk['title']}\n\n")
            f.write(f"{chunk['content']}\n\n---\n\n")
    
    print(f"Saved combined chunks file: {combined_filepath}")
    saved_files.append(str(combined_filepath))
    
    return saved_files


async def process_section(
    input_file: str,
    output_dir: str,
    verbose: bool = True
) -> Dict:
    """Process a single section file and generate chunks."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print(f"{'='*60}\n")
    
    # Read the markdown file
    section_title, section_content = read_markdown_file(input_file)
    
    if verbose:
        print(f"Section title: {section_title}")
        print(f"Content length: {len(section_content)} characters")
        print()
    
    # Create the prompt
    user_prompt = create_user_prompt(section_title, section_content)
    
    # Generate chunks using LLM
    response = await generate_chunks(user_prompt)
    
    # Extract chunks from response
    chunks = extract_chunks_from_response(response)
    
    if not chunks:
        print("Warning: No chunks extracted from response. Raw response saved for debugging.")
        # Save raw response for debugging
        debug_dir = Path(output_dir) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_file = debug_dir / f"raw_response_{Path(input_file).stem}.txt"
        with open(debug_file, 'w') as f:
            f.write(response)
        return {
            'input_file': input_file,
            'section_title': section_title,
            'chunks': [],
            'saved_files': [],
            'error': 'No chunks extracted'
        }
    
    if verbose:
        print(f"\nExtracted {len(chunks)} chunks")
    
    # Save chunks
    output_path = Path(output_dir)
    saved_files = save_chunks(chunks, output_path, section_title)
    
    return {
        'input_file': input_file,
        'section_title': section_title,
        'chunks': chunks,
        'saved_files': saved_files
    }


async def process_directory(
    input_dir: str,
    output_dir: str,
    verbose: bool = True
) -> List[Dict]:
    """Process all markdown files in a directory."""
    input_path = Path(input_dir)
    results = []
    
    # Find all markdown files
    md_files = list(input_path.glob("*.md"))
    
    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return results
    
    print(f"Found {len(md_files)} markdown files to process")
    
    for md_file in md_files:
        result = await process_section(
            str(md_file),
            output_dir,
            verbose=verbose
        )
        results.append(result)
    
    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Split academic paper sections into semantic chunks using NVIDIA LLM"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to a single markdown file to process"
    )
    parser.add_argument(
        "--input-dir", "-d",
        type=str,
        default=None,
        help="Path to directory containing markdown files to process"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for chunks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")
    
    if args.input:
        result = await process_section(
            args.input,
            args.output_dir,
            verbose=args.verbose
        )
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Section: {result['section_title']}")
        print(f"Chunks created: {len(result['chunks'])}")
        print(f"Files saved: {len(result['saved_files'])}")
        
    elif args.input_dir:
        results = await process_directory(
            args.input_dir,
            args.output_dir,
            verbose=args.verbose
        )
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for result in results:
            print(f"Section: {result['section_title']}")
            print(f"  Chunks created: {len(result['chunks'])}")
            print(f"  Files saved: {len(result['saved_files'])}")


if __name__ == "__main__":
    asyncio.run(main())


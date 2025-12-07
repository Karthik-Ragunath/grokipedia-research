"""
Code-Chunk Aligner

This script takes paper chunks and code files, then uses an LLM to find
relevant code sections for each chunk, creating aligned chunks-with-code.

Usage:
    python code_chunk_aligner.py \\
        --chunks-dir ../../deepseek-moe-explainer/chunks \\
        --code-dir ../../deepseek-moe-explainer/deepseek-code \\
        --output-dir ../../deepseek-moe-explainer/chunks-with-code
"""

import asyncio
import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


# System prompt for code-chunk alignment
SYSTEM_PROMPT = """You are an expert at analyzing academic papers and their corresponding code implementations. Your task is to find relevant code sections that implement or relate to concepts described in a paper chunk.

Guidelines:
1. Identify code that directly implements the concept described in the chunk
2. Look for class definitions, functions, or code blocks that correspond to the theory
3. Consider both high-level structure and low-level implementation details
4. Include relevant imports and helper functions if they're essential to understanding
5. Prioritize the most relevant code sections - don't include everything

Output Format:
Return your analysis in the following XML format:

<alignment>
<relevance_score>0-10</relevance_score>
<explanation>Brief explanation of how the code relates to the chunk concept</explanation>
<code_sections>
<section file="filename.py" start_line="X" end_line="Y">
Brief description of what this code section does
</section>
... more sections if needed ...
</code_sections>
</alignment>

Important:
- relevance_score: 0 = no relevant code, 10 = perfect implementation match
- Only include code sections with score >= 5
- Limit to 3-5 most relevant code sections
- Be precise with line numbers
- If no relevant code exists, set relevance_score to 0 and leave code_sections empty"""


def read_file(filepath: str) -> str:
    """Read file content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def read_code_files(code_dir: str, extensions: List[str] = ['.py']) -> Dict[str, Tuple[str, List[str]]]:
    """
    Read all code files from a directory.
    Returns dict: {filename: (content, lines_list)}
    """
    code_files = {}
    code_path = Path(code_dir)
    
    for ext in extensions:
        for filepath in code_path.glob(f'**/*{ext}'):
            rel_path = filepath.relative_to(code_path)
            content = read_file(str(filepath))
            lines = content.split('\n')
            code_files[str(rel_path)] = (content, lines)
    
    return code_files


def read_chunks(chunks_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Read all chunk files from the chunks directory.
    Returns dict: {section_name: {chunk_filename: content}}
    """
    chunks = {}
    chunks_path = Path(chunks_dir)
    
    for section_dir in chunks_path.iterdir():
        if section_dir.is_dir():
            section_name = section_dir.name
            chunks[section_name] = {}
            
            for chunk_file in section_dir.glob('chunk_*.md'):
                content = read_file(str(chunk_file))
                chunks[section_name][chunk_file.name] = content
    
    return chunks


def create_code_summary(code_files: Dict[str, Tuple[str, List[str]]]) -> str:
    """Create a summary of available code files with key classes/functions."""
    summary_parts = []
    
    for filename, (content, lines) in code_files.items():
        summary_parts.append(f"\n### File: {filename}")
        summary_parts.append(f"Lines: {len(lines)}")
        
        # Extract class and function definitions
        classes = []
        functions = []
        
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('class '):
                match = re.match(r'\s*class\s+(\w+)', line)
                if match:
                    classes.append(f"  - Line {i}: class {match.group(1)}")
            elif line.strip().startswith('def '):
                match = re.match(r'\s*def\s+(\w+)', line)
                if match:
                    # Only top-level or important functions
                    indent = len(line) - len(line.lstrip())
                    if indent <= 4:  # Top level or class method
                        functions.append(f"  - Line {i}: def {match.group(1)}")
        
        if classes:
            summary_parts.append("Classes:")
            summary_parts.extend(classes[:20])  # Limit to first 20
        if functions:
            summary_parts.append("Functions:")
            summary_parts.extend(functions[:30])  # Limit to first 30
    
    return '\n'.join(summary_parts)


def create_user_prompt(chunk_content: str, chunk_title: str, code_files: Dict[str, Tuple[str, List[str]]]) -> str:
    """Create the user prompt for code-chunk alignment."""
    
    # Create code summary
    code_summary = create_code_summary(code_files)
    
    # Create full code content (limited to avoid token limits)
    code_content_parts = []
    for filename, (content, lines) in code_files.items():
        # Limit each file to first 500 lines for context
        limited_lines = lines[:500]
        numbered_lines = [f"{i+1:4d}| {line}" for i, line in enumerate(limited_lines)]
        code_content_parts.append(f"\n=== {filename} ===\n")
        code_content_parts.append('\n'.join(numbered_lines))
        if len(lines) > 500:
            code_content_parts.append(f"\n... ({len(lines) - 500} more lines)")
    
    code_content = '\n'.join(code_content_parts)
    
    return f"""Find relevant code sections that implement or relate to this paper concept:

## Paper Chunk: {chunk_title}

{chunk_content}

---

## Available Code Structure:
{code_summary}

---

## Code Content (with line numbers):
{code_content}

---

Analyze the paper chunk and identify the most relevant code sections that implement or demonstrate the concepts described. Focus on finding direct implementations of the mathematical formulas, algorithms, or architectural components mentioned."""


async def align_chunk_with_code(
    chunk_content: str,
    chunk_title: str,
    code_files: Dict[str, Tuple[str, List[str]]]
) -> Dict:
    """Use LLM to find relevant code for a chunk."""
    
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    user_prompt = create_user_prompt(chunk_content, chunk_title, code_files)
    
    print(f"  Calling LLM for alignment...")
    
    completion = client.chat.completions.create(
        model="qwen/qwen3-coder-480b-a35b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        top_p=0.8,
        max_tokens=4096,
        stream=True
    )
    
    full_content = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            full_content += chunk.choices[0].delta.content
    
    return parse_alignment_response(full_content, code_files)


def parse_alignment_response(response: str, code_files: Dict[str, Tuple[str, List[str]]]) -> Dict:
    """Parse the LLM alignment response."""
    result = {
        'relevance_score': 0,
        'explanation': '',
        'code_sections': []
    }
    
    # Extract relevance score
    score_match = re.search(r'<relevance_score>(\d+)</relevance_score>', response)
    if score_match:
        result['relevance_score'] = int(score_match.group(1))
    
    # Extract explanation
    explanation_match = re.search(r'<explanation>(.*?)</explanation>', response, re.DOTALL)
    if explanation_match:
        result['explanation'] = explanation_match.group(1).strip()
    
    # Extract code sections
    section_pattern = r'<section\s+file="([^"]+)"\s+start_line="(\d+)"\s+end_line="(\d+)">\s*(.*?)\s*</section>'
    sections = re.findall(section_pattern, response, re.DOTALL)
    
    for filename, start_line, end_line, description in sections:
        start = int(start_line)
        end = int(end_line)
        
        # Get the actual code
        code_content = ""
        if filename in code_files:
            _, lines = code_files[filename]
            # Adjust for 0-indexing
            code_lines = lines[max(0, start-1):min(len(lines), end)]
            code_content = '\n'.join(code_lines)
        
        result['code_sections'].append({
            'file': filename,
            'start_line': start,
            'end_line': end,
            'description': description.strip(),
            'code': code_content
        })
    
    return result


def create_aligned_chunk(
    original_content: str,
    alignment: Dict
) -> str:
    """Create a new chunk file with aligned code."""
    
    output_parts = [original_content]
    
    if alignment['relevance_score'] >= 5 and alignment['code_sections']:
        output_parts.append("\n\n---\n")
        output_parts.append(f"\n## Corresponding Code Implementation\n")
        output_parts.append(f"\n**Relevance Score:** {alignment['relevance_score']}/10\n")
        output_parts.append(f"\n**Explanation:** {alignment['explanation']}\n")
        
        for i, section in enumerate(alignment['code_sections'], 1):
            output_parts.append(f"\n### Code Section {i}: {section['description']}\n")
            output_parts.append(f"\n**File:** `{section['file']}` (lines {section['start_line']}-{section['end_line']})\n")
            output_parts.append(f"\n```python\n{section['code']}\n```\n")
    else:
        output_parts.append("\n\n---\n")
        output_parts.append("\n*No directly corresponding code implementation found for this concept.*\n")
    
    return ''.join(output_parts)


async def process_chunks(
    chunks_dir: str,
    code_dir: str,
    output_dir: str,
    verbose: bool = True
) -> Dict:
    """Process all chunks and create aligned versions."""
    
    print(f"\n{'='*70}")
    print("CODE-CHUNK ALIGNER")
    print(f"{'='*70}")
    print(f"\nChunks directory: {chunks_dir}")
    print(f"Code directory: {code_dir}")
    print(f"Output directory: {output_dir}")
    
    # Read code files
    print("\n[STEP 1] Reading code files...")
    code_files = read_code_files(code_dir)
    print(f"  Found {len(code_files)} code files:")
    for filename in code_files:
        _, lines = code_files[filename]
        print(f"    - {filename}: {len(lines)} lines")
    
    # Read chunks
    print("\n[STEP 2] Reading chunks...")
    chunks = read_chunks(chunks_dir)
    total_chunks = sum(len(section_chunks) for section_chunks in chunks.values())
    print(f"  Found {total_chunks} chunks across {len(chunks)} sections")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each chunk
    print("\n[STEP 3] Aligning chunks with code...")
    results = {}
    
    for section_name, section_chunks in chunks.items():
        print(f"\n  Section: {section_name}")
        section_output = output_path / section_name
        section_output.mkdir(parents=True, exist_ok=True)
        
        results[section_name] = {}
        
        for chunk_filename, chunk_content in sorted(section_chunks.items()):
            print(f"    Processing: {chunk_filename}")
            
            # Extract title from chunk
            title_match = re.match(r'^#\s+(.+)$', chunk_content, re.MULTILINE)
            chunk_title = title_match.group(1) if title_match else chunk_filename
            
            # Get alignment from LLM
            alignment = await align_chunk_with_code(
                chunk_content, 
                chunk_title, 
                code_files
            )
            
            print(f"      Relevance: {alignment['relevance_score']}/10, Sections: {len(alignment['code_sections'])}")
            
            # Create aligned chunk
            aligned_content = create_aligned_chunk(chunk_content, alignment)
            
            # Save to output
            output_file = section_output / chunk_filename
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(aligned_content)
            
            results[section_name][chunk_filename] = {
                'relevance_score': alignment['relevance_score'],
                'num_sections': len(alignment['code_sections'])
            }
        
        # Copy _all_chunks.md if it exists
        all_chunks_src = Path(chunks_dir) / section_name / "_all_chunks.md"
        if all_chunks_src.exists():
            shutil.copy(all_chunks_src, section_output / "_all_chunks.md")
    
    # Print summary
    print(f"\n{'='*70}")
    print("ALIGNMENT COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput saved to: {output_dir}")
    
    print("\nResults summary:")
    for section_name, section_results in results.items():
        print(f"\n  {section_name}:")
        for chunk_name, result in section_results.items():
            score = result['relevance_score']
            sections = result['num_sections']
            print(f"    {chunk_name}: score={score}/10, code_sections={sections}")
    
    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Align paper chunks with corresponding code implementations"
    )
    parser.add_argument(
        "--chunks-dir", "-c",
        type=str,
        required=True,
        help="Directory containing paper chunks"
    )
    parser.add_argument(
        "--code-dir", "-d",
        type=str,
        required=True,
        help="Directory containing code files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for aligned chunks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    await process_chunks(
        chunks_dir=args.chunks_dir,
        code_dir=args.code_dir,
        output_dir=args.output_dir,
        verbose=args.verbose
    )


if __name__ == "__main__":
    asyncio.run(main())


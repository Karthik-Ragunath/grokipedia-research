"""
Paper Chunking Pipeline

A complete pipeline that:
1. Extracts sections from a LaTeX file
2. Splits sections into semantic chunks using NVIDIA LLM
3. Stores everything in organized directories

Usage:
    python paper_chunking_pipeline.py \\
        --latex-file path/to/main.tex \\
        --sections "Introduction,Preliminaries,Architecture" \\
        --output-dir path/to/output

Example for DeepSeek-MoE paper:
    python paper_chunking_pipeline.py \\
        --latex-file ../deepseek-moe-explainer/deepseek-moe-paper/main.tex \\
        --sections "Introduction,Preliminaries: Mixture-of-Experts for Transformers,DeepSeekMoE Architecture" \\
        --output-dir ../deepseek-moe-explainer
"""

import asyncio
import argparse
from pathlib import Path
from typing import List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent))

from latex_section_extractor import extract_sections_from_latex
from semantic_chunker import process_section, process_directory


async def run_pipeline(
    latex_file: str,
    section_names: List[str],
    output_dir: str,
    skip_chunking: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Run the complete paper chunking pipeline.
    
    Args:
        latex_file: Path to the LaTeX file
        section_names: List of section names to extract
        output_dir: Base output directory
        skip_chunking: If True, only extract sections without LLM chunking
        verbose: Print verbose output
    
    Returns:
        Dictionary with pipeline results
    """
    output_path = Path(output_dir)
    
    # Create output directories
    extracted_dir = output_path / "extracted_sections"
    chunks_dir = output_path / "chunks"
    
    extracted_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("PAPER CHUNKING PIPELINE")
    print(f"{'='*70}")
    print(f"\nLaTeX file: {latex_file}")
    print(f"Sections to extract: {section_names}")
    print(f"Output directory: {output_dir}")
    print(f"\n{'='*70}")
    
    # Step 1: Extract sections from LaTeX
    print("\n[STEP 1] Extracting sections from LaTeX file...")
    print("-" * 50)
    
    extracted_files = extract_sections_from_latex(
        latex_file=latex_file,
        section_names=section_names,
        output_dir=str(extracted_dir)
    )
    
    print(f"\nExtracted {len(extracted_files)} sections:")
    for name, filepath in extracted_files.items():
        print(f"  - {name}: {filepath}")
    
    if skip_chunking:
        print("\n[SKIPPING] LLM chunking step (--skip-chunking flag set)")
        return {
            'extracted_files': extracted_files,
            'chunks': {}
        }
    
    # Step 2: Split sections into semantic chunks
    print(f"\n{'='*70}")
    print("[STEP 2] Splitting sections into semantic chunks using LLM...")
    print("-" * 50)
    
    chunk_results = {}
    for section_name, filepath in extracted_files.items():
        print(f"\nProcessing section: {section_name}")
        result = await process_section(
            input_file=filepath,
            output_dir=str(chunks_dir),
            verbose=verbose
        )
        chunk_results[section_name] = result
    
    # Print summary
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\nExtracted sections saved to: {extracted_dir}")
    print(f"Semantic chunks saved to: {chunks_dir}")
    
    print("\nResults summary:")
    for section_name, result in chunk_results.items():
        num_chunks = len(result.get('chunks', []))
        print(f"  - {section_name}: {num_chunks} chunks")
    
    return {
        'extracted_files': extracted_files,
        'chunks': chunk_results
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline for extracting and chunking paper sections"
    )
    parser.add_argument(
        "--latex-file", "-f",
        type=str,
        required=True,
        help="Path to the LaTeX file"
    )
    parser.add_argument(
        "--sections", "-s",
        type=str,
        required=True,
        help="Comma-separated list of section names to extract"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Base output directory for all generated files"
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Only extract sections, skip LLM chunking step"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    section_names = [s.strip() for s in args.sections.split(",")]
    
    await run_pipeline(
        latex_file=args.latex_file,
        section_names=section_names,
        output_dir=args.output_dir,
        skip_chunking=args.skip_chunking,
        verbose=args.verbose
    )


if __name__ == "__main__":
    asyncio.run(main())


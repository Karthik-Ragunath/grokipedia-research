"""
LaTeX Section Extractor

This script extracts sections from a LaTeX file based on section names
and stores them as individual markdown files.

Usage:
    python latex_section_extractor.py --file main.tex --sections "Introduction,Preliminaries,Architecture"
    python latex_section_extractor.py --file main.tex --sections "Introduction" --output-dir ./extracted
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional


def read_latex_file(filepath: str) -> str:
    """Read the content of a LaTeX file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def extract_balanced_braces(text: str, start: int) -> tuple[str, int]:
    """
    Extract content within balanced braces starting from position start.
    Returns (content, end_position).
    """
    if text[start] != '{':
        raise ValueError(f"Expected '{{' at position {start}")
    
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start+1:i], i + 1
        i += 1
    
    # Unbalanced braces - return what we have
    return text[start+1:], len(text)


def find_section_boundaries(content: str) -> List[Dict]:
    """
    Find all section boundaries in the LaTeX content.
    Returns a list of dicts with section name, start position, and end position.
    """
    # Pattern to find \section or \section*
    section_start_pattern = r'\\section\*?\{'
    
    sections = []
    for match in re.finditer(section_start_pattern, content):
        start_pos = match.start()
        brace_start = match.end() - 1  # Position of the opening brace
        
        # Extract the section name with balanced braces
        section_name, header_end = extract_balanced_braces(content, brace_start)
        
        sections.append({
            'name': section_name,
            'raw_name': section_name,
            'start': start_pos,
            'header_end': header_end,
            'end': None  # Will be filled in next step
        })
    
    # Set end positions (each section ends where the next one begins, or at EOF)
    for i, section in enumerate(sections):
        if i + 1 < len(sections):
            section['end'] = sections[i + 1]['start']
        else:
            section['end'] = len(content)
    
    return sections


def normalize_section_name(name: str) -> str:
    """Normalize section name for comparison (lowercase, strip whitespace, remove LaTeX commands)."""
    # Handle specific known commands first
    name = re.sub(r'\\spmoe\{\}', 'DeepSeekMoE', name, flags=re.IGNORECASE)
    
    # Remove common LaTeX commands with content
    name = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', name)  # \command{text} -> text
    
    # Remove commands without braces
    name = re.sub(r'\\[a-zA-Z]+', '', name)  # \command -> ''
    
    name = name.lower().strip()
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name)
    return name


def extract_section(content: str, sections: List[Dict], target_section: str) -> Optional[Dict]:
    """
    Extract a specific section from the LaTeX content.
    Uses fuzzy matching to find the section.
    """
    target_normalized = normalize_section_name(target_section)
    
    for section in sections:
        section_normalized = normalize_section_name(section['name'])
        
        # Check for exact match or partial match
        if (target_normalized == section_normalized or 
            target_normalized in section_normalized or
            section_normalized in target_normalized):
            
            section_content = content[section['header_end']:section['end']]
            return {
                'name': section['name'],
                'content': section_content.strip(),
                'full_content': content[section['start']:section['end']].strip()
            }
    
    return None


def latex_to_markdown(latex_content: str) -> str:
    """
    Convert LaTeX content to Markdown format.
    This is a basic conversion - handles common LaTeX constructs.
    """
    text = latex_content
    
    # Handle labels (remove them)
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    
    # Handle subsections
    text = re.sub(r'\\subsection\*?\{([^}]+)\}', r'\n## \1\n', text)
    text = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'\n### \1\n', text)
    
    # Handle paragraph headers
    text = re.sub(r'\\paragraph\{([^}]+)\}', r'\n**\1**\n', text)
    
    # Handle text formatting
    text = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', text)
    text = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', text)
    text = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', text)
    text = re.sub(r'\\underline\{([^}]+)\}', r'_\1_', text)
    
    # Handle custom commands (like \spmoe{})
    text = re.sub(r'\\spmoe\{\}', 'DeepSeekMoE', text)
    
    # Handle citations
    text = re.sub(r'~?\\citep?\{([^}]+)\}', r'[\1]', text)
    text = re.sub(r'~?\\citet?\{([^}]+)\}', r'[\1]', text)
    
    # Handle references
    text = re.sub(r'\\ref\{([^}]+)\}', r'(ref: \1)', text)
    text = re.sub(r'Figure~?\\ref\{([^}]+)\}', r'Figure (ref: \1)', text)
    text = re.sub(r'Section~?\\ref\{([^}]+)\}', r'Section (ref: \1)', text)
    
    # Handle footnotes (convert to inline)
    text = re.sub(r'\\footnote\{([^}]+)\}', r' (footnote: \1)', text)
    
    # Handle itemize/enumerate environments
    text = re.sub(r'\\begin\{itemize\}', '', text)
    text = re.sub(r'\\end\{itemize\}', '', text)
    text = re.sub(r'\\begin\{enumerate\}', '', text)
    text = re.sub(r'\\end\{enumerate\}', '', text)
    text = re.sub(r'\\item\s*', '\n- ', text)
    
    # Handle math environments (keep them as-is with $$ for block math)
    text = re.sub(r'\\begin\{align\*?\}', '\n$$\n', text)
    text = re.sub(r'\\end\{align\*?\}', '\n$$\n', text)
    text = re.sub(r'\\begin\{equation\*?\}', '\n$$\n', text)
    text = re.sub(r'\\end\{equation\*?\}', '\n$$\n', text)
    
    # Handle figures (extract caption)
    def replace_figure(match):
        figure_content = match.group(0)  # Get the entire match
        caption_match = re.search(r'\\caption\{([^}]+)\}', figure_content)
        if caption_match:
            return f'\n[Figure: {caption_match.group(1)}]\n'
        return '\n[Figure]\n'
    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', replace_figure, text, flags=re.DOTALL)
    
    # Clean up extra commands
    text = re.sub(r'\\centering', '', text)
    text = re.sub(r'\\includegraphics\[[^\]]*\]\{[^}]+\}', '', text)
    
    # Handle special characters
    text = text.replace('~', ' ')
    text = text.replace('\\%', '%')
    text = text.replace('\\$', '$')
    text = text.replace('\\&', '&')
    text = text.replace('\\_', '_')
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def save_section_to_markdown(section_data: Dict, output_dir: Path, include_header: bool = True) -> str:
    """Save extracted section to a markdown file."""
    # Create a safe filename from section name
    safe_name = re.sub(r'[^\w\s-]', '', section_data['name'])
    safe_name = re.sub(r'\s+', '_', safe_name).lower()
    filename = f"{safe_name}.md"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    # Convert content to markdown
    md_content = latex_to_markdown(section_data['content'])
    
    # Add header
    if include_header:
        full_content = f"# {section_data['name']}\n\n{md_content}"
    else:
        full_content = md_content
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print(f"Saved section '{section_data['name']}' to {filepath}")
    return str(filepath)


def extract_sections_from_latex(
    latex_file: str,
    section_names: List[str],
    output_dir: str = None
) -> Dict[str, str]:
    """
    Main function to extract multiple sections from a LaTeX file.
    
    Args:
        latex_file: Path to the LaTeX file
        section_names: List of section names to extract
        output_dir: Directory to save extracted sections (default: same dir as latex file)
    
    Returns:
        Dictionary mapping section names to their output file paths
    """
    latex_path = Path(latex_file)
    
    if output_dir is None:
        output_path = latex_path.parent / "extracted_sections"
    else:
        output_path = Path(output_dir)
    
    # Read LaTeX content
    content = read_latex_file(latex_file)
    
    # Find all sections
    sections = find_section_boundaries(content)
    print(f"Found {len(sections)} sections in {latex_file}:")
    for s in sections:
        print(f"  - {s['name']}")
    
    # Extract requested sections
    results = {}
    for section_name in section_names:
        section_data = extract_section(content, sections, section_name)
        if section_data:
            filepath = save_section_to_markdown(section_data, output_path)
            results[section_data['name']] = filepath
        else:
            print(f"Warning: Section '{section_name}' not found in {latex_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract sections from a LaTeX file and save as Markdown"
    )
    parser.add_argument(
        "--file", "-f",
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
        default=None,
        help="Output directory for extracted sections"
    )
    
    args = parser.parse_args()
    
    section_names = [s.strip() for s in args.sections.split(",")]
    
    results = extract_sections_from_latex(
        latex_file=args.file,
        section_names=section_names,
        output_dir=args.output_dir
    )
    
    print(f"\nExtracted {len(results)} sections:")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()


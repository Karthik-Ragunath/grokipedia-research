import { NextRequest, NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join, resolve } from 'path'
import { existsSync } from 'fs'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ chunkName: string }> }
) {
  try {
    const { chunkName } = await params
    const PROJECT_ROOT = resolve(process.cwd(), '..')
    
    // Construct the markdown file path
    // chunkName is like "chunk_01_introduction_to_grpo"
    // We need "chunk_01_introduction_to_grpo_summary.md"
    const summaryPath = join(
      PROJECT_ROOT,
      'paper-manim-viz-explanations/grpo-explainer/chunk-summary/reinforcement_learning',
      `${chunkName}_summary.md`
    )
    
    if (!existsSync(summaryPath)) {
      return NextResponse.json({ error: 'Content not found' }, { status: 404 })
    }
    
    const content = await readFile(summaryPath, 'utf-8')
    
    return NextResponse.json({ content })
  } catch (error) {
    console.error('Error reading GRPO content:', error)
    return NextResponse.json({ error: 'Failed to read content' }, { status: 500 })
  }
}


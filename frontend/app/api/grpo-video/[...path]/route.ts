import { NextRequest, NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join, resolve } from 'path'
import { existsSync } from 'fs'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const { path: pathArray } = await params
    const PROJECT_ROOT = resolve(process.cwd(), '..')
    
    // Reconstruct the path from the array
    const relativePath = pathArray.join('/')
    const videoPath = join(
      PROJECT_ROOT,
      'paper-manim-viz-explanations/grpo-explainer/generated_videos/reinforcement_learning',
      relativePath,
      'video.mp4'
    )
    
    if (!existsSync(videoPath)) {
      return NextResponse.json({ error: 'Video not found' }, { status: 404 })
    }
    
    const videoBuffer = await readFile(videoPath)
    
    return new NextResponse(videoBuffer, {
      headers: {
        'Content-Type': 'video/mp4',
        'Content-Length': videoBuffer.length.toString(),
        'Cache-Control': 'public, max-age=31536000, immutable',
      },
    })
  } catch (error) {
    console.error('Error serving GRPO video:', error)
    return NextResponse.json({ error: 'Failed to serve video' }, { status: 500 })
  }
}


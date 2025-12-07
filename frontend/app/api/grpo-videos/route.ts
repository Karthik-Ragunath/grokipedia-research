import { NextResponse } from 'next/server'
import { readdir, stat } from 'fs/promises'
import { join, resolve } from 'path'
import { existsSync } from 'fs'

// Resolve paths relative to the project root (one level up from frontend)
const PROJECT_ROOT = resolve(process.cwd(), '..')
const GRPO_VIDEOS_DIR = join(PROJECT_ROOT, 'paper-manim-viz-explanations/grpo-explainer/generated_videos/reinforcement_learning')

interface VideoItem {
  id: string
  title: string
  description: string
  videoPath: string
  chunkNumber: number
  header: string
  chunkDir?: string // Optional: for matching with markdown files
}

// Recursively find all video.mp4 files
async function findVideoFiles(dir: string, basePath: string = ''): Promise<Array<{ path: string, relativePath: string }>> {
  const videos: Array<{ path: string, relativePath: string }> = []
  
  try {
    const entries = await readdir(dir, { withFileTypes: true })
    
    for (const entry of entries) {
      const fullPath = join(dir, entry.name)
      const relativePath = basePath ? join(basePath, entry.name) : entry.name
      
      if (entry.isDirectory()) {
        // Recursively search in subdirectories
        const subVideos = await findVideoFiles(fullPath, relativePath)
        videos.push(...subVideos)
      } else if (entry.name === 'video.mp4') {
        videos.push({ path: fullPath, relativePath })
      }
    }
  } catch (error) {
    console.error(`Error reading directory ${dir}:`, error)
  }
  
  return videos
}

export async function GET() {
  try {
    if (!existsSync(GRPO_VIDEOS_DIR)) {
      console.error(`GRPO_VIDEOS_DIR does not exist: ${GRPO_VIDEOS_DIR}`)
      return NextResponse.json({ error: `GRPO videos directory not found: ${GRPO_VIDEOS_DIR}` }, { status: 500 })
    }
    
    const videoItems: VideoItem[] = []
    
    // Find all video.mp4 files recursively
    const videoFiles = await findVideoFiles(GRPO_VIDEOS_DIR)
    
    // Sort by path to maintain consistent order
    videoFiles.sort((a, b) => a.relativePath.localeCompare(b.relativePath))
    
    for (const videoFile of videoFiles) {
      // Extract chunk info from path
      // e.g., chunk_01_introduction_to_grpo/video.mp4
      // or chunk_08_training_and_evaluation_setup/reinforcement_learning/chunk_08_training_and_evaluation_setup/video.mp4
      const pathParts = videoFile.relativePath.split('/')
      const chunkDir = pathParts[0] // First directory is the chunk directory
      
      // Extract chunk number and name
      const match = chunkDir.match(/chunk_(\d+)_(.+)/)
      if (!match) continue
      
      const chunkNumber = parseInt(match[1], 10)
      const chunkName = match[2]
      
      // Format title from chunk name
      const title = chunkName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
      
      // Use the relative path for the API endpoint
      // Remove 'video.mp4' and use the directory path
      const apiPath = videoFile.relativePath.replace(/\/video\.mp4$/, '')
      
      videoItems.push({
        id: `grpo_chunk_${chunkNumber}`,
        title,
        description: `GRPO: ${title}`,
        videoPath: `/api/grpo-video/${apiPath}`,
        chunkNumber,
        header: chunkName,
        chunkDir: chunkDir, // Store chunk directory name for markdown lookup
      })
    }
    
    return NextResponse.json({ videos: videoItems })
  } catch (error) {
    console.error('Error loading GRPO videos:', error)
    return NextResponse.json({ error: 'Failed to load GRPO videos' }, { status: 500 })
  }
}


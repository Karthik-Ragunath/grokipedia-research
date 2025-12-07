# Grokipedia Mobile Frontend

Mobile-optimized frontend for the Grokipedia research paper visualization application.

## Features

- Mobile-first responsive design
- Touch-optimized UI components
- Video playback with LaTeX rendering
- Paper submission via ArXiv URL or PDF upload

## Development

```bash
# Install dependencies
npm install

# Run development server (port 3001)
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Deployment on Vercel

This application is configured for Vercel deployment:

1. Connect your repository to Vercel
2. Set the root directory to `frontend-mobile`
3. Vercel will automatically detect Next.js and use the correct build settings

Alternatively, you can deploy from the root directory by configuring:
- Root Directory: `frontend-mobile`
- Build Command: `npm install && npm run build`
- Output Directory: `.next`

## Project Structure

```
frontend-mobile/
├── app/
│   ├── api/          # API routes
│   ├── layout.tsx    # Root layout
│   ├── page.tsx      # Home page
│   └── globals.css   # Global styles
├── components/
│   ├── ui/           # UI components
│   └── paper-submission-page.tsx
└── lib/
    └── utils.ts      # Utility functions
```

## Mobile Optimizations

- Single-column layout for small screens
- Larger touch targets (minimum 44x44px)
- Optimized video player for mobile
- Responsive modal dialogs
- Touch-friendly interactions


import type { NextConfig } from 'next'
import os from 'os'
import path from 'path'

const base = process.env.NEXT_PUBLIC_BASE_PATH || ''
// Store Next build artifacts (.next) outside OneDrive to avoid Windows/OneDrive readlink issues
const distDir = process.env.NEXT_DIST_DIR || path.join(os.tmpdir(), 'stock_dashboard_next')

// Explicit workspace root to silence Next.js warning about multiple lockfiles
const root = path.resolve(__dirname, '..')

// Allow CI (GitHub Pages) to request a static export build
const isExport = process.env.NEXT_OUTPUT_EXPORT === '1'

const nextConfig: NextConfig = {
  // Server runtime on Vercel by default; CI can flip to static export
  basePath: base || undefined,
  assetPrefix: base || undefined,
  distDir,
  outputFileTracingRoot: root,
  ...(isExport ? { output: 'export' } : {}),
}

export default nextConfig


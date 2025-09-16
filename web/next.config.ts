import type { NextConfig } from 'next'
import os from 'os'
import path from 'path'

const base = process.env.NEXT_PUBLIC_BASE_PATH || ''
// Store Next build artifacts (.next) outside OneDrive to avoid Windows/OneDrive readlink issues
const distDir = process.env.NEXT_DIST_DIR || path.join(os.tmpdir(), 'stock_dashboard_next')

const nextConfig: NextConfig = {
  output: 'export',
  basePath: base || undefined,
  assetPrefix: base || undefined,
  distDir,
}

export default nextConfig

const base = process.env.NEXT_PUBLIC_BASE_PATH || ''
export const BASE = base
export const DATA_BASE = process.env.NEXT_PUBLIC_DATA_BASE || `${base}/data`

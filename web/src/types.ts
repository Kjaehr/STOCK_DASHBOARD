export type StockMeta = {
  generated_at?: string
  tickers: string[]
}

export type BuyZone = {
  type: 'sma_pullback' | 'breakout_retest' | string
  ma?: string
  price_low: number
  price_high: number
  confidence?: number
  rationale?: string
}

export type ExitLevels = {
  stop_suggest?: number | null
  targets?: number[]
  rationale?: string
}

export type PositionHealth = {
  entry_readiness?: number
  exit_risk?: number
  in_buy_zone?: boolean
  dist_to_stop_pct?: number | null
  dist_to_t1_pct?: number | null
}

export type StockData = {
  ticker: string
  score?: number
  fund_points?: number
  tech_points?: number
  sent_points?: number
  price?: number | null
  sma50?: number | null
  sma200?: number | null
  updated_at?: string
  flags?: string[]
  fundamentals?: Record<string, any>
  technicals?: Record<string, any>
  sentiment?: Record<string, any>
  buy_zones?: BuyZone[]
  exit_levels?: ExitLevels | null
  position_health?: PositionHealth | null
}


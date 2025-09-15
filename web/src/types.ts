export type StockMeta = {
  generated_at?: string
  tickers: string[]
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
}


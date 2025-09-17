import type { Metadata } from 'next'
import TickerFromQuery from './FromQueryClient'

export const metadata: Metadata = {
  title: 'Ticker - Stock Dashboard',
}

export default function Page() {
  return <TickerFromQuery />
}


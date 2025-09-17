import type { Metadata } from 'next'
import TickerClient from './[id]/Client'
import { BASE } from '../../base'

export const metadata: Metadata = {
  title: 'Ticker - Stock Dashboard',
}

export default function Page(props: any) {
  const idParam = props?.searchParams?.id as string | undefined
  const id = idParam ? decodeURIComponent(idParam) : ''
  if (!id) {
    return (
      <section style={{padding:16}}>
        <h1 style={{margin:'8px 0'}}>Missing ticker</h1>
        <p>Please provide a ticker id, e.g. /ticker?id=MSFT</p>
        <a href={`${BASE}/`}>&larr; Back</a>
      </section>
    )
  }
  return <TickerClient id={id} />
}


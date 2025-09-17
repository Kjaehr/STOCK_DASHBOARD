"use client"
import { useSearchParams } from 'next/navigation'
import TickerClient from './[id]/Client'
import { BASE } from '../../base'

export default function TickerFromQuery() {
  const sp = useSearchParams()
  const idParam = sp?.get('id') || ''
  const id = idParam ? decodeURIComponent(idParam) : ''

  if (!id) {
    return (
      <section style={{ padding: 16 }}>
        <h1 style={{ margin: '8px 0' }}>Missing ticker</h1>
        <p>Please provide a ticker id, e.g. /ticker?id=MSFT</p>
        <a href={`${BASE}/`}>&larr; Back</a>
      </section>
    )
  }

  return <TickerClient id={id} />
}


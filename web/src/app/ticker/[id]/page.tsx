"use client"
import { useEffect, useState } from 'react'
import type { StockData } from '../../../types'

export default function TickerPage({ params }: { params: { id: string } }) {
  const id = decodeURIComponent(params.id)
  const [data, setData] = useState<StockData | null>(null)

  useEffect(() => {
    const run = async () => {
      try {
        const json = await fetch(`/data/${id.replace(/\s+/g,'_')}.json`).then(r => r.json())
        setData(json)
      } catch {}
    }
    run()
  }, [id])

  return (
    <section>
      <h1 style={{margin:'8px 0'}}>{id}</h1>
      <pre style={{background:'#fafafa', padding:12, overflow:'auto'}}>{JSON.stringify(data, null, 2)}</pre>
    </section>
  )
}


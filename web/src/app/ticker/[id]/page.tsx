import type { Metadata } from "next"
import path from "path"
import fs from "fs/promises"
import TickerClient from "./Client"

const DATA_META_PATH = path.join(process.cwd(), "..", "data", "meta.json")

async function readTickers(): Promise<string[]> {
  try {
    const raw = await fs.readFile(DATA_META_PATH, "utf8")
    const json = JSON.parse(raw)
    if (Array.isArray(json?.tickers)) return json.tickers as string[]
  } catch (e) {
    console.warn('ticker meta read failed', e)
  }
  return []
}

export async function generateStaticParams() {
  const tickers = await readTickers()
  return tickers.map(t => ({ id: encodeURIComponent(t) }))
}

export async function generateMetadata({ params }: { params: { id: string } }): Promise<Metadata> {
  const decoded = decodeURIComponent(params.id)
  return {
    title: `${decoded} - Stock Dashboard`,
  }
}

export default function Page({ params }: { params: { id: string } }) {
  const id = decodeURIComponent(params.id)
  return <TickerClient id={id} />
}

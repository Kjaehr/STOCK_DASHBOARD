"use client"
import { Chart as ChartJS, LineElement, PointElement, LinearScale, CategoryScale, Filler, Legend, Tooltip, TimeScale } from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Filler, Legend, Tooltip, TimeScale)

type BuyZone = { price_low: number; price_high: number; type?: string; ma?: string; confidence?: number; rationale?: string }

const baseOptions: any = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  elements: { point: { radius: 0 } },
  plugins: {
    legend: {
      display: true,
      position: 'top' as const,
      labels: {
        filter: (legendItem: any, chart: any) => {
          const ds = chart?.data?.datasets?.[legendItem.datasetIndex] as any
          // Hide overlay-only datasets (zones, stops, targets) from legend to reduce clutter
          return !(ds && (ds.isOverlay === true || ds.isZoneOverlay === true))
        },
      },
    },
    tooltip: {
      mode: 'index' as const,
      intersect: false,
      filter: (item: any) => {
        const ds: any = item?.dataset
        // Suppress the "low" edge dataset from tooltip; show only the high/zone label
        if (ds && ds.isZoneOverlay && ds.role === 'low') return false
        return true
      },
      callbacks: {
        label: (ctx: any) => {
          const ds: any = ctx.dataset
          if (ds && ds.isZoneOverlay) {
            const lo = ds.zoneLow
            const hi = ds.zoneHigh
            const name = ds.zoneLabel || 'Buy zone'
            return `${name}: ${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(lo)}–${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(hi)}`
          }
          const v = ctx.parsed.y
          if (v == null || Number.isNaN(v)) return `${ctx.dataset.label}: —`
          return `${ctx.dataset.label}: ${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(v)}`
        },
      },
    },
  },
  scales: {
    x: { display: false },
    y: {
      ticks: { maxTicksLimit: 4 },
      grid: { color: 'rgba(0,0,0,0.05)' },
    },
  },
  spanGaps: true,
}

export function PriceChart({ labels, close, sma50, sma200, zones, exits }: { labels: string[]; close: (number|null)[]; sma50: (number|null)[]; sma200: (number|null)[]; zones?: BuyZone[]; exits?: { stop_suggest?: number | null; targets?: number[] } }) {
  const overlay: any[] = []
  if (zones && zones.length && labels.length) {
    zones.forEach((z) => {
      if (typeof z.price_low !== 'number' || typeof z.price_high !== 'number') return
      const lo = z.price_low, hi = z.price_high
      const base = new Array(labels.length).fill(null).map(() => hi)
      const low = new Array(labels.length).fill(null).map(() => lo)
      const color = z.type === 'breakout_retest' ? 'rgba(253, 224, 71, 0.25)' /* amber-300/40 */ : 'rgba(72, 187, 120, 0.20)' /* green-400/30 */
      const border = z.type === 'breakout_retest' ? 'rgba(217, 119, 6, 0.6)' : 'rgba(56, 161, 105, 0.6)'
      const label = z.type === 'sma_pullback' ? `Buy zone (${z.ma ?? 'SMA'})` : (z.type === 'breakout_retest' ? 'Buy zone (Retest)' : 'Buy zone')
      // Upper edge
      overlay.push({
        label,
        data: base,
        borderColor: border,
        backgroundColor: color,
        fill: { target: `-1` }, // fill to previous dataset (the low edge)
        tension: 0.00001,
        borderWidth: 0,
        pointRadius: 0,
        order: 0,
        isZoneOverlay: true,
        isOverlay: true,
        role: 'high',
        zoneLow: lo,
        zoneHigh: hi,
        zoneLabel: label,
      })
      // Lower edge (invisible; acts as fill target)
      overlay.push({
        label: label + ' (low)',
        data: low,
        borderColor: border,
        backgroundColor: color,
        fill: false,
        tension: 0.00001,
        borderWidth: 0,
        pointRadius: 0,
        order: 0,
        isZoneOverlay: true,
        isOverlay: true,
        role: 'low',
        zoneLow: lo,
        zoneHigh: hi,
        zoneLabel: label,
      })
    })
  }
  // Exit overlays: stop (red) and targets (purple dashed)
  if (labels.length && exits) {
    const stop = exits.stop_suggest
    if (typeof stop === 'number' && Number.isFinite(stop)) {
      overlay.push({
        label: 'Stop', data: new Array(labels.length).fill(stop), borderColor: '#e53e3e', borderWidth: 1.5, pointRadius: 0, tension: 0, order: 10, isOverlay: true,
      })
    }
    const targets = Array.isArray(exits.targets) ? exits.targets : []
    targets.forEach((t, i) => {
      if (typeof t !== 'number' || !Number.isFinite(t)) return
      overlay.push({
        label: `Target ${i+1}`, data: new Array(labels.length).fill(t), borderColor: '#6b46c1', borderDash: [6,4], borderWidth: 1.3, pointRadius: 0, tension: 0, order: 10, isOverlay: true,
      })
    })
  }
  const data = {
    labels,
    datasets: [
      ...overlay,
      { label: 'Close', data: close, borderColor: '#2b6cb0', tension: 0.2 },
      { label: 'SMA50', data: sma50, borderColor: '#38a169', tension: 0.2 },
      { label: 'SMA200', data: sma200, borderColor: '#dd6b20', tension: 0.2 },
    ],
  }
  return <Line data={data} options={baseOptions} />
}

export function LineChart({ labels, data: values, label, color, ySuggestedMin, ySuggestedMax }: {
  labels: string[]; data: (number|null)[]; label: string; color: string; ySuggestedMin?: number; ySuggestedMax?: number;
}) {
  const data = { labels, datasets: [ { label, data: values, borderColor: color, tension: 0.2 } ] }
  const opts = {
    ...baseOptions,
    scales: {
      ...baseOptions.scales,
      y: { ...baseOptions.scales.y, suggestedMin: ySuggestedMin, suggestedMax: ySuggestedMax },
    },
  }
  return <Line data={data} options={opts} />
}


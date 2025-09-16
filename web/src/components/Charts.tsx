"use client"
import { Chart as ChartJS, LineElement, PointElement, LinearScale, CategoryScale, Filler, Legend, Tooltip, TimeScale } from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Filler, Legend, Tooltip, TimeScale)

const baseOptions: any = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  elements: { point: { radius: 0 } },
  plugins: {
    legend: { display: true, position: 'top' as const },
    tooltip: {
      mode: 'index' as const,
      intersect: false,
      callbacks: {
        label: (ctx: any) => {
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

export function PriceChart({ labels, close, sma50, sma200 }: { labels: string[]; close: (number|null)[]; sma50: (number|null)[]; sma200: (number|null)[] }) {
  const data = {
    labels,
    datasets: [
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


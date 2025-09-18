"use client"
import { Suspense } from "react"
import Leaderboard from "../components/Leaderboard"

export default function Page() {
  return (
    <Suspense fallback={<section className="p-4 text-sm text-muted-foreground">Loading…</section>}>
      <Leaderboard />
    </Suspense>
  )
}

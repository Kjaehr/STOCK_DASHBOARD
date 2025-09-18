"use client"
import { ThemeToggle } from "../theme-toggle"
import { Input } from "../ui/input"
import { Button } from "../ui/button"

export function Topbar({ right }: { right?: React.ReactNode }){
  return (
    <header className="sticky top-0 z-40 border-b bg-background/60 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto max-w-7xl px-4 h-14 flex items-center gap-3">
        <div className="flex-1 flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 text-sm text-muted-foreground">
            <span>Search</span>
          </div>
          <div className="w-[300px] max-w-[40vw]">
            <Input placeholder="Search tickers... (Ctrl+/)" className="h-9" />
          </div>
        </div>
        <div className="flex items-center gap-2">
          {right}
          <ThemeToggle />
          <Button variant="outline" size="sm" asChild>
            <a href="https://github.com/Kjaehr/STOCK_DASHBOARD" target="_blank" rel="noreferrer">GitHub</a>
          </Button>
        </div>
      </div>
    </header>
  )
}


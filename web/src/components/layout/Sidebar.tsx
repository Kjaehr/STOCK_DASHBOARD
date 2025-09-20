"use client"
import { BASE } from "../../base"
import { usePathname } from "next/navigation"

const navItems = [
  { href: `${BASE}/`, label: "Dashboard", key: "dashboard", emoji: "📊" },
  { href: `${BASE}/`, label: "Leaderboard", key: "leaderboard", emoji: "🏆" },
  { href: `${BASE}/backtest`, label: "Backtest", key: "backtest", emoji: "🧪" },
  { href: `${BASE}/portfolio`, label: "Portfolio", key: "portfolio", emoji: "💼" },
  { href: `${BASE}/alerts`, label: "Alerts", key: "alerts", emoji: "🔔", soon: true },
  { href: `${BASE}/settings`, label: "Settings", key: "settings", emoji: "⚙️", soon: true },
]

export function Sidebar() {
  const pathname = usePathname() || "";
  function isActive(itemHref: string) {
    // Treat "/" and "/leaderboard" as same home page in this app
    if (itemHref === `${BASE}/`) return pathname === `${BASE}` || pathname === `${BASE}/` || pathname.startsWith(`${BASE}/ticker`) || pathname === `/${""}`
    return pathname.startsWith(itemHref)
  }
  return (
    <aside className="hidden md:flex w-64 shrink-0 flex-col border-r bg-background/60 backdrop-blur">
      <div className="h-14 border-b flex items-center gap-2 px-4">
        <Logo />
        <div className="font-semibold tracking-tight">Stock Dashboard</div>
      </div>
      <nav className="flex-1 overflow-y-auto p-2 space-y-1">
        {navItems.map((item)=>{
          const active = isActive(item.href)
          return (
            <a key={item.key} href={item.href}
               className={`group flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors border ${active?"bg-accent/40 border-accent":"hover:bg-muted/40 border-transparent"}`}>
              <span className="text-lg leading-none">{item.emoji}</span>
              <span className="flex-1">{item.label}</span>
              {item.soon && <span className="text-[10px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground">soon</span>}
            </a>
          )
        })}
      </nav>
      <div className="p-3 text-xs text-muted-foreground">
        © {new Date().getFullYear()}
      </div>
    </aside>
  )
}

function Logo(){
  return (
    <div className="h-8 w-8 rounded-md bg-gradient-to-br from-violet-500 to-cyan-400 p-[2px]">
      <div className="h-full w-full rounded-[6px] bg-background/80 flex items-center justify-center">
        <svg viewBox="0 0 24 24" width={16} height={16} className="text-foreground/80">
          <path d="M4 13l4-4 3 3 5-5 4 4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
    </div>
  )
}


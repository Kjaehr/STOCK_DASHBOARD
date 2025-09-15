export const metadata = {
  title: 'Stock Dashboard',
  description: 'Free-tier stock screener and portfolio (static export)',
}

import './globals.css'

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header style={{padding:'12px 16px', borderBottom:'1px solid #eee'}}>
          <nav style={{display:'flex', gap:16}}>
            <a href="/">Leaderboard</a>
            <a href="/portfolio">Portfolio</a>
          </nav>
        </header>
        <main style={{padding:'16px'}}>{children}</main>
      </body>
    </html>
  )
}


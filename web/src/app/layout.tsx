export const metadata = {
  title: 'Stock Dashboard',
  description: 'Free-tier stock screener and portfolio (static export)',
}

import './globals.css'
import { BASE } from '../base'
import { ThemeProvider } from "../components/theme-provider"

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <header className="border-b border-border">
            <nav className="container mx-auto flex gap-4 px-4 py-3">
              <a className="hover:underline" href={`${BASE}/`}>Leaderboard</a>
              <a className="hover:underline" href={`${BASE}/portfolio`}>Portfolio</a>
            </nav>
          </header>
          <main className="container mx-auto px-4 py-4">{children}</main>
        </ThemeProvider>
      </body>
    </html>
  )
}


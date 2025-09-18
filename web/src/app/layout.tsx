export const metadata = {
  title: 'Stock Dashboard',
  description: 'Free-tier stock screener and portfolio (static export)',
}

import './globals.css'
import { ThemeProvider } from "../components/theme-provider"
import { DashboardShell } from "../components/layout/DashboardShell"

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground">
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
          <DashboardShell>
            {children}
          </DashboardShell>
        </ThemeProvider>
      </body>
    </html>
  )
}

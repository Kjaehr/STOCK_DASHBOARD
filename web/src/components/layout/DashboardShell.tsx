"use client"
import React from "react"
import { Sidebar } from "./Sidebar"
import { Topbar } from "./Topbar"

export function DashboardShell({ children, right }: { children: React.ReactNode; right?: React.ReactNode }){
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex w-full">
        <Sidebar />
        <div className="flex-1 min-w-0">
          <Topbar right={right} />
          <div className="mx-auto max-w-7xl p-4">
            {children}
          </div>
        </div>
      </div>
    </div>
  )
}


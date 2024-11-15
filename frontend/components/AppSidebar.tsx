"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname, useRouter, useSearchParams } from "next/navigation"
import { 
  Activity, 
  Home,
  Eye, 
  Droplets,
  Brain,
  Timer,
  LineChart,
  Calendar,
  MessageSquare,
  Settings,
  Icon
} from "lucide-react"
import { cn } from "@/lib/utils"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarSeparator,
  SidebarMenuButton,
  useSidebar
} from "@/components/ui/sidebar"
import Logo from "@/components/ui/logo"

export function AppSidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const searchParams = useSearchParams()
  const { state } = useSidebar()

  const isDemo = pathname === "/demo"

  const mainNavItems = [
    {
      title: "Dashboard",
      href: "/sbs-pro/dashboard",
      Icon: Home,
      description: "Overview of your health metrics"
    },
    {
      title: "Services",
      href: "/sbs-pro/services",
      Icon: Settings,
      description: "Configure monitoring services"
    },
    {
      title: "Analytics",
      href: "/sbs-pro/analytics",
      Icon: LineChart,
      description: "Detailed health data analysis"
    },
    {
      title: "Activity",
      href: "/sbs-pro/activity",
      Icon: Activity,
      description: "Your daily activities and goals"
    }
  ]

  const featureNavItems = [
    {
      title: "Posture Monitor",
      href: "/sbs-pro/posture",
      Icon: Brain,
      color: "text-emerald-500",
      description: "Real-time posture tracking"
    },
    {
      title: "Eye Care",
      href: "/sbs-pro/eye-care",
      Icon: Eye,
      color: "text-blue-500",
      description: "Blink rate monitoring"
    },
    {
      title: "Hydration",
      href: "/sbs-pro/hydration",
      Icon: Droplets,
      color: "text-cyan-500",
      description: "Water intake tracking"
    },
    {
      title: "AI Assistant",
      href: "/sbs-pro/chatbot",
      Icon: MessageSquare,
      color: "text-violet-500",
      description: "Your health companion"
    }
  ]

  const toolsNavItems = [
    {
      title: "Timer",
      href: "/sbs-pro/timer",
      Icon: Timer,
      color: "text-purple-500",
      description: "Break reminders"
    },
    {
      title: "Calendar",
      href: "/sbs-pro/calendar",
      Icon: Calendar,
      color: "text-orange-500",
      description: "Schedule your health routine"
    }
  ]

  return (
    <Sidebar className="hidden md:flex">
      <SidebarHeader className="relative border-b border-sidebar-border/50 p-4">
        <div className="flex items-center gap-3">
          <Logo className="h-10 w-10" />
          <Link href="/" className={cn("font-display text-xl font-black tracking-tight", 
            state === "collapsed" ? "hidden" : "block")}>
            <span className="bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
              SitBlinkSip
            </span>
          </Link>
        </div>
      </SidebarHeader>

      <SidebarContent className={cn("p-4 transition-all duration-200", 
        state === "collapsed" ? "w-[70px]" : "w-[240px]")}>
        <SidebarGroup>
          <SidebarGroupLabel className="text-sm font-semibold text-gray-500 pl-4">
            Main
          </SidebarGroupLabel>
          {mainNavItems.map((item) => (
            <SidebarMenuButton
              key={item.href}
              isActive={isDemo ? searchParams.get("view") === item.href.slice(1) : pathname === item.href}
              tooltip={item.description}
              className={cn(
                "text-base mb-2 pl-4 transition-all duration-200 relative",
                "hover:before:absolute hover:before:right-0 hover:before:top-0 hover:before:w-1 hover:before:h-full hover:before:bg-blue-400/50",
                (isDemo ? searchParams.get("view") === item.href.slice(1) : pathname === item.href) && "before:absolute before:right-0 before:top-0 before:w-1 before:h-full before:bg-blue-600",
                ((isDemo ? searchParams.get("view") === item.href.slice(1) : pathname === item.href)
                  ? "bg-blue-50 text-blue-600 font-medium"
                  : "hover:bg-gray-50"
                )
              )}
              onClick={() => {
                if (isDemo) {
                  const params = new URLSearchParams(searchParams)
                  params.set("view", item.href.slice(1))
                  router.push(`/demo?${params.toString()}`, { scroll: false })
                } else {
                  router.push(item.href)
                }
              }}
            >
              <div className="flex items-center gap-2">
                <item.Icon className={cn("h-5 w-5", 
                  (isDemo ? searchParams.get("view") === item.href.slice(1) : pathname === item.href) 
                    ? "text-blue-600" 
                    : "text-gray-500"
                )} />
                <span className={state === "collapsed" ? "hidden" : "block"}>{item.title}</span>
              </div>
            </SidebarMenuButton>
          ))}
        </SidebarGroup>

        <SidebarSeparator className="my-4" />

        <SidebarGroup>
          <SidebarGroupLabel className="text-sm font-semibold text-gray-500 pl-4">
            Features
          </SidebarGroupLabel>
          {featureNavItems.map((item) => (
            <SidebarMenuButton
              key={item.href}
              isActive={isDemo ? searchParams.get("view") === item.href.slice(1) : pathname === item.href}
              tooltip={item.description}
              className={cn(
                "text-base mb-2 pl-4 transition-all duration-200 relative",
                "hover:before:absolute hover:before:right-0 hover:before:top-0 hover:before:w-1 hover:before:h-full hover:before:bg-blue-400/50",
                (isDemo ? searchParams.get("view") === item.href.slice(1) : pathname === item.href) && "before:absolute before:right-0 before:top-0 before:w-1 before:h-full before:bg-blue-600",
                ((isDemo ? searchParams.get("view") === item.href.slice(1) : pathname === item.href)
                  ? "bg-blue-50 text-blue-600 font-medium"
                  : "hover:bg-gray-50"
                )
              )}
              onClick={() => {
                if (isDemo) {
                  const params = new URLSearchParams(searchParams)
                  params.set("view", item.href.slice(1))
                  router.push(`/demo?${params.toString()}`, { scroll: false })
                } else {
                  router.push(item.href)
                }
              }}
            >
              <div className="flex items-center gap-2">
                <item.Icon className={cn("h-5 w-5", item.color)} />
                <span className={state === "collapsed" ? "hidden" : "block"}>{item.title}</span>
              </div>
            </SidebarMenuButton>
          ))}
        </SidebarGroup>

        <SidebarSeparator className="my-4" />

        <SidebarGroup>
          <SidebarGroupLabel className="text-sm font-semibold text-gray-500 pl-4">
            Tools
          </SidebarGroupLabel>
          {toolsNavItems.map((item) => (
            <SidebarMenuButton
              key={item.href}
              isActive={isDemo ? searchParams.get("view") === item.href.slice(1) : pathname === item.href}
              tooltip={item.description}
              className={cn(
                "text-base mb-2 pl-4 transition-all duration-200 relative",
                "hover:before:absolute hover:before:right-0 hover:before:top-0 hover:before:w-1 hover:before:h-full hover:before:bg-blue-400/50",
                (isDemo ? searchParams.get("view") === item.href.slice(1) : pathname === item.href) && "before:absolute before:right-0 before:top-0 before:w-1 before:h-full before:bg-blue-600",
                ((isDemo ? searchParams.get("view") === item.href.slice(1) : pathname === item.href)
                  ? "bg-blue-50 text-blue-600 font-medium"
                  : "hover:bg-gray-50"
                )
              )}
              onClick={() => {
                if (isDemo) {
                  const params = new URLSearchParams(searchParams)
                  params.set("view", item.href.slice(1))
                  router.push(`/demo?${params.toString()}`, { scroll: false })
                } else {
                  router.push(item.href)
                }
              }}
            >
              <div className="flex items-center gap-2">
                <item.Icon className={cn("h-5 w-5", item.color)} />
                <span className={state === "collapsed" ? "hidden" : "block"}>{item.title}</span>
              </div>
            </SidebarMenuButton>
          ))}
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-sidebar-border/50 p-4">
        <div className="flex items-center justify-between">
          <div className={cn("flex items-center gap-3", 
            state === "collapsed" ? "justify-center w-full" : "")}>
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
              <span className="text-sm font-medium text-blue-700">IS</span>
            </div>
            <div className={cn("flex flex-col", state === "collapsed" ? "hidden" : "block")}>
              <span className="text-sm font-medium">Ishwor Subedi</span>
             
            </div>
          </div>
          <button 
            className={cn("rounded-lg p-2 hover:bg-gray-100", 
              state === "collapsed" ? "hidden" : "block")}
            onClick={() => router.push('/settings')}
          >
            <Settings className="h-5 w-5 text-gray-500" />
          </button>
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <AppSidebar />
      <main className="flex-1 overflow-y-auto">
        <div className="container p-6">
          {children}
        </div>
      </main>
    </div>
  );
}
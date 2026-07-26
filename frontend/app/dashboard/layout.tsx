import { SidebarProvider } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/AppSidebar"
import { PersonProvider } from "./PersonContext"
import { MonitoringProvider } from "./MonitoringContext"

export default function DemoLayout({ children }: { children: React.ReactNode }) {
  return (
    <PersonProvider>
      <MonitoringProvider>
        <SidebarProvider>
          <div className="flex min-h-screen w-full">
            <AppSidebar />
            <main className="flex-1 overflow-y-auto custom-scrollbar">
              {children}
            </main>
          </div>
        </SidebarProvider>
      </MonitoringProvider>
    </PersonProvider>
  )
}
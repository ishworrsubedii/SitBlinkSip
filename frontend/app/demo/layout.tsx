import { SidebarProvider, SidebarTrigger, Sidebar, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/AppSidebar"

export default function DemoLayout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider defaultOpen={true}>
      <div className="flex min-h-screen">
        <AppSidebar />
        <SidebarInset className="flex-1">
          <div className="flex h-16 items-center justify-between gap-4 border-b border-border px-6">
            <SidebarTrigger />
            <div className="flex items-center gap-4">
              {/* Add any header content here */}
            </div>
          </div>
          <div className="p-6">
            {children}
          </div>
        </SidebarInset>
      </div>
    </SidebarProvider>
  )
} 
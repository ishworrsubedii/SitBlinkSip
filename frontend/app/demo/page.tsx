"use client";

import { useSearchParams } from "next/navigation";
import DashboardPage from "./dashboard/page";
import WellnessMonitor from "./services/page";

export default function DemoPage() {
  const searchParams = useSearchParams();
  const view = searchParams.get("view") || "dashboard";

  return (
    <div className="flex-1 w-full h-full overflow-y-auto">
      {view === "dashboard" && <DashboardPage />}
      {view === "services" && <WellnessMonitor />}
    </div>
  );
}

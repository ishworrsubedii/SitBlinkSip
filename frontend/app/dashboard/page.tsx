"use client";

import { useSearchParams } from "next/navigation";
import DashboardPage from "./overview/page";
import WellnessMonitor from "./services/page";
import AnalyticsPage from "./analytics/page";

export default function DemoPage() {
  const searchParams = useSearchParams();
  const view = searchParams.get("view") || "dashboard";

  return (
    <div className="h-screen overflow-hidden">
      <div className="h-full overflow-y-auto">
        {view === "dashboard" && <DashboardPage />}
        {view === "services" && <WellnessMonitor />}
        {view === "analytics" && <AnalyticsPage />}
      </div>
    </div>
  );
}

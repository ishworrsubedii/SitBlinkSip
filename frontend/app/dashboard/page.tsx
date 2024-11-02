"use client";

import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, Eye, Droplets, Play, Square, Timer } from "lucide-react";
import { Progress } from "@/components/ui/progress";

export default function DashboardPage() {
  // Add loading state
  const [isClient, setIsClient] = React.useState(false);

  // Use useEffect to handle client-side mounting
  React.useEffect(() => {
    setIsClient(true);
  }, []);

  // Demo states - replace with real data later
  const [postureActive, setPostureActive] = React.useState(false);
  const [eyeBlinkActive, setEyeBlinkActive] = React.useState(false);
  const [waterBreakActive, setWaterBreakActive] = React.useState(false);

  // Return loading state or null while client-side rendering isn't ready
  if (!isClient) {
    return null; // Or a loading spinner
  }

  return (
    <div className="container space-y-8 p-8">
      {/* Stats Section */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Posture Score</CardTitle>
            <Brain className="h-4 w-4 text-emerald-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">87%</div>
            <Progress value={87} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">
              +2.5% from last session
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Blink Rate</CardTitle>
            <Eye className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12 bpm</div>
            <Progress value={60} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">
              Normal range: 10-20 bpm
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Hydration</CardTitle>
            <Droplets className="h-4 w-4 text-cyan-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">750ml</div>
            <Progress value={45} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">
              Target: 2000ml daily
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Control Panels */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-emerald-500" />
              Posture Monitor
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button 
              className="w-full"
              variant={postureActive ? "destructive" : "default"}
              onClick={() => setPostureActive(!postureActive)}
            >
              {postureActive ? (
                <><Square className="mr-2 h-4 w-4" /> Stop Monitoring</>
              ) : (
                <><Play className="mr-2 h-4 w-4" /> Start Monitoring</>
              )}
            </Button>
            <div className="text-sm text-muted-foreground">
              Last correction: 5 minutes ago
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5 text-blue-500" />
              Eye Care Monitor
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button 
              className="w-full"
              variant={eyeBlinkActive ? "destructive" : "default"}
              onClick={() => setEyeBlinkActive(!eyeBlinkActive)}
            >
              {eyeBlinkActive ? (
                <><Square className="mr-2 h-4 w-4" /> Stop Monitoring</>
              ) : (
                <><Play className="mr-2 h-4 w-4" /> Start Monitoring</>
              )}
            </Button>
            <div className="text-sm text-muted-foreground">
              Next break in: 15 minutes
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Droplets className="h-5 w-5 text-cyan-500" />
              Water Break Timer
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button 
              className="w-full"
              variant={waterBreakActive ? "destructive" : "default"}
              onClick={() => setWaterBreakActive(!waterBreakActive)}
            >
              {waterBreakActive ? (
                <><Square className="mr-2 h-4 w-4" /> Stop Timer</>
              ) : (
                <><Play className="mr-2 h-4 w-4" /> Start Timer</>
              )}
            </Button>
            <div className="text-sm text-muted-foreground">
              Next reminder in: 30 minutes
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Additional Stats or Charts could go here */}
      <Card>
        <CardHeader>
          <CardTitle>Today's Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">9:00 AM</span>
              <span>Started monitoring</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">10:30 AM</span>
              <span>Water break reminder</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">11:15 AM</span>
              <span>Posture correction</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
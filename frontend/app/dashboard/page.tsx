"use client";

import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, Eye, Droplets, Play, Square, Timer, Settings, Activity, BrainCircuit } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Brush, Legend, ReferenceLine 
} from 'recharts';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";

// Add time range presets
const timePresets = {
  '1m': 60 * 1000,
  '5m': 5 * 60 * 1000,
  '15m': 15 * 60 * 1000,
  '30m': 30 * 60 * 1000,
  '1h': 60 * 60 * 1000,
};

// Time range options with more detailed configuration
const timeRangeOptions = [
  { value: 'minute', label: 'Per Minute', interval: 60 * 1000 },
  { value: 'hourly', label: 'Hourly', interval: 60 * 60 * 1000 },
  { value: 'daily', label: 'Daily', interval: 24 * 60 * 60 * 1000 },
  { value: 'weekly', label: 'Weekly', interval: 7 * 24 * 60 * 60 * 1000 },
  { value: 'monthly', label: 'Monthly', interval: 30 * 24 * 60 * 60 * 1000 },
];

export default function DashboardPage() {
  // Add loading state
  const [isClient, setIsClient] = React.useState(false);

  // Use useEffect to handle client-side mounting
  React.useEffect(() => {
    setIsClient(true);
  }, []);

  // Monitoring states
  const [isMonitoring, setIsMonitoring] = React.useState(false);
  const [selectedFeatures, setSelectedFeatures] = React.useState({
    posture: false,
    eyeBlink: false
  });

  // Settings states
  const [settings, setSettings] = React.useState({
    posture: {
      threshold: 80,
      saveImages: false
    },
    eyeBlink: {
      threshold: 12,
      saveImages: false
    },
    water: {
      interval: 30,
      enabled: false
    }
  });

  // Add this sample data near your other state declarations
  const [mockData] = React.useState({
    blinkData: [
      { time: '09:00:00', blinked: true, timestamp: new Date('2024-03-20T09:00:00').getTime() },
      { time: '09:00:01', blinked: false, timestamp: new Date('2024-03-20T09:00:01').getTime() },
      { time: '09:00:02', blinked: true, timestamp: new Date('2024-03-20T09:00:02').getTime() },
      { time: '09:00:03', blinked: false, timestamp: new Date('2024-03-20T09:00:03').getTime() },
      // Add more data points as needed
    ],
    postureData: [
      { time: '09:00:00', posture: 'good', score: 95, timestamp: new Date('2024-03-20T09:00:00').getTime() },
      { time: '09:00:30', posture: 'bad', score: 65, timestamp: new Date('2024-03-20T09:00:30').getTime() },
      { time: '09:01:00', posture: 'good', score: 90, timestamp: new Date('2024-03-20T09:01:00').getTime() },
      { time: '09:01:30', posture: 'good', score: 92, timestamp: new Date('2024-03-20T09:01:30').getTime() },
      // Add more data points as needed
    ],
  });

  // Add time range state
  const [timeRange, setTimeRange] = React.useState('hourly');
  const [notifications, setNotifications] = React.useState(false);

  // Function to format timestamp to readable time
  const formatXAxis = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Return loading state or null while client-side rendering isn't ready
  if (!isClient) {
    return null; // Or a loading spinner
  }

  // Add time range handler
  const handleTimeRangeChange = (value: string) => {
    setTimeRange(value);
    // Add logic to fetch data for the selected time range
    const selectedInterval = timeRangeOptions.find(opt => opt.value === value)?.interval || 0;
    const now = Date.now();
    // Update your data fetching logic here
  };

  const handleNotificationsChange = (checked: boolean) => {
    setNotifications(checked);
    // Add notification handling logic
  };

  // Add these calculations before the return statement
  const totalBlinks = mockData.blinkData.filter(d => d.blinked).length;
  const avgPostureScore = Math.round(
    mockData.postureData.reduce((acc, curr) => acc + curr.score, 0) / mockData.postureData.length
  );

  return (
    <div className="container space-y-6 p-8">
      {/* Top Stats Section */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="bg-primary/5">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Current Posture</CardTitle>
            <Brain className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">87%</div>
            <Progress value={87} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">
              Good posture detected
            </p>
          </CardContent>
        </Card>

        <Card className="bg-primary/5">
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

        <Card className="bg-primary/5">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Session Duration</CardTitle>
            <Timer className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1h 23m</div>
            <Progress value={70} className="mt-2 bg-primary/20" />
            <p className="text-xs text-muted-foreground mt-2">
              Target: 2h daily
            </p>
          </CardContent>
        </Card>

        <Card className="bg-primary/5">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Water Intake</CardTitle>
            <Droplets className="h-4 w-4 text-cyan-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">750ml</div>
            <Progress value={45} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">
              Next break in 15m
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Analysis Settings Section */}
      <Card className="bg-blue-50/50 backdrop-blur-sm border border-blue-100">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div className="flex items-center gap-2">
            <BrainCircuit className="h-5 w-5 text-blue-600" />
            <CardTitle>Analysis Controls</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-2">
            {/* Time Range Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-blue-900/70">Analysis Period</label>
              <Select value={timeRange} onValueChange={handleTimeRangeChange}>
                <SelectTrigger className="w-full bg-white border-blue-100">
                  <SelectValue placeholder="Select time range" />
                </SelectTrigger>
                <SelectContent>
                  {timeRangeOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Analysis Stats */}
            <div className="flex items-center justify-end gap-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">{totalBlinks}</p>
                <p className="text-sm text-blue-900/70">Total Blinks</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">{avgPostureScore}%</p>
                <p className="text-sm text-blue-900/70">Avg Posture</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Graphs Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Blink Rate Analysis */}
        <Card className="backdrop-blur-sm bg-white/95 border border-blue-100">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div className="flex items-center gap-2">
              <Eye className="h-4 w-4 text-blue-600" />
              <CardTitle className="text-lg font-medium">Blink Rate Analysis</CardTitle>
            </div>
            <div className="text-sm text-blue-900/70">
              {`${totalBlinks} blinks`}
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockData.blinkData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <defs>
                    <linearGradient id="blinkGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2563eb" stopOpacity={0.1}/>
                      <stop offset="95%" stopColor="#2563eb" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-20" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatXAxis}
                    type="number"
                    fontSize={11}
                    stroke="#64748b"
                  />
                  <YAxis 
                    domain={[0, 30]} 
                    fontSize={11}
                    stroke="#64748b"
                    label={{ value: 'Blinks/min', angle: -90, position: 'insideLeft', fontSize: 11 }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '6px',
                      border: '1px solid rgba(37, 99, 235, 0.1)',
                      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="blinkRate" 
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={false}
                    fill="url(#blinkGradient)"
                  />
                  <ReferenceLine 
                    y={settings.eyeBlink.threshold} 
                    stroke="#dc2626" 
                    strokeDasharray="3 3"
                    label={{ value: 'Threshold', position: 'right', fontSize: 11 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Posture Analysis */}
        <Card className="backdrop-blur-sm bg-white/95 border border-blue-100">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-blue-600" />
              <CardTitle className="text-lg font-medium">Posture Analysis</CardTitle>
            </div>
            <div className="text-sm text-blue-900/70">
              {`${avgPostureScore}% average`}
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockData.postureData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <defs>
                    <linearGradient id="postureGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2563eb" stopOpacity={0.1}/>
                      <stop offset="95%" stopColor="#2563eb" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-20" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatXAxis}
                    type="number"
                    fontSize={11}
                    stroke="#64748b"
                  />
                  <YAxis 
                    domain={[0, 100]} 
                    fontSize={11}
                    stroke="#64748b"
                    label={{ value: 'Posture Score %', angle: -90, position: 'insideLeft', fontSize: 11 }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '6px',
                      border: '1px solid rgba(37, 99, 235, 0.1)',
                      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="score" 
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={false}
                    fill="url(#postureGradient)"
                  />
                  <ReferenceLine 
                    y={settings.posture.threshold} 
                    stroke="#dc2626" 
                    strokeDasharray="3 3"
                    label={{ value: 'Threshold', position: 'right', fontSize: 11 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
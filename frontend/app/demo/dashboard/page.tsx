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
  Tooltip, ResponsiveContainer, Brush, Legend, ReferenceLine, ComposedChart 
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
  { value: '1h', label: 'Last Hour', interval: 60 * 60 * 1000 },
  { value: '2h', label: 'Last 2 Hours', interval: 2 * 60 * 60 * 1000 },
  { value: '4h', label: 'Last 4 Hours', interval: 4 * 60 * 60 * 1000 },
  { value: '8h', label: 'Last 8 Hours', interval: 8 * 60 * 60 * 1000 },
  { value: '12h', label: 'Last 12 Hours', interval: 12 * 60 * 60 * 1000 },
  { value: '24h', label: 'Last 24 Hours', interval: 24 * 60 * 60 * 1000 },
];

// Add day filter options
const getDayFilterOptions = () => {
  const options = [];
  for (let i = 0; i < 7; i++) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    const formattedDate = date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric'
    });
    const value = date.toISOString().split('T')[0];
    options.push({
      value,
      label: i === 0 ? `Today (${formattedDate})` : formattedDate,
      isToday: i === 0
    });
  }
  return options;
};

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
      { time: '09:00:00', blinkRate: 15, timestamp: new Date('2024-03-20T09:00:00').getTime() },
      { time: '09:00:01', blinkRate: 12, timestamp: new Date('2024-03-20T09:00:01').getTime() },
      { time: '09:00:02', blinkRate: 18, timestamp: new Date('2024-03-20T09:00:02').getTime() },
      { time: '09:00:03', blinkRate: 14, timestamp: new Date('2024-03-20T09:00:03').getTime() },
      // Add more data points as needed
    ],
    postureData: [
      { time: '09:00:00', posture: 'good', score: 95, timestamp: new Date('2024-03-20T09:00:00').getTime() },
      { time: '09:00:30', posture: 'bad', score: 65, timestamp: new Date('2024-03-20T09:00:30').getTime() },
      { time: '09:01:00', posture: 'good', score: 90, timestamp: new Date('2024-03-20T09:01:00').getTime() },
      { time: '09:01:30', posture: 'good', score: 92, timestamp: new Date('2024-03-20T09:01:30').getTime() },
      // Add more data points as needed
    ],
    waterData: [
      { time: '09:00', amount: 250, goal: 300, timestamp: new Date('2024-03-20T09:00:00').getTime() },
      { time: '11:00', amount: 200, goal: 300, timestamp: new Date('2024-03-20T11:00:00').getTime() },
      { time: '13:00', amount: 300, goal: 300, timestamp: new Date('2024-03-20T13:00:00').getTime() },
      { time: '15:00', amount: 150, goal: 300, timestamp: new Date('2024-03-20T15:00:00').getTime() },
      { time: '17:00', amount: 250, goal: 300, timestamp: new Date('2024-03-20T17:00:00').getTime() },
    ],
  });

  // Add time range state
  const [timeRange, setTimeRange] = React.useState('hourly');
  const [notifications, setNotifications] = React.useState(false);

  // Add day filter state
  const [dayFilter, setDayFilter] = React.useState('today');

  // Function to format timestamp to readable time
  const formatXAxis = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString('en-US', { 
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
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

  // Add this calculation before the return statement
  const totalBlinks = mockData.blinkData.reduce((sum, data) => sum + data.blinkRate, 0);

  const avgPostureScore = Math.round(
    mockData.postureData.reduce((sum, data) => sum + data.score, 0) / mockData.postureData.length
  );

  return (
    <div className="w-full space-y-6 p-6">
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
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-blue-900/70">Select Date</label>
              <Select value={dayFilter} onValueChange={setDayFilter}>
                <SelectTrigger className="w-full bg-white border-blue-100">
                  <SelectValue placeholder="Select date" />
                </SelectTrigger>
                <SelectContent className="bg-white backdrop-blur-sm">
                  {getDayFilterOptions().map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {dayFilter === getDayFilterOptions()[0].value && (
              <div className="space-y-2">
                <label className="text-sm font-medium text-blue-900/70">Time Window</label>
                <Select value={timeRange} onValueChange={handleTimeRangeChange}>
                  <SelectTrigger className="w-full bg-white border-blue-100">
                    <SelectValue placeholder="Select time window" />
                  </SelectTrigger>
                  <SelectContent className="bg-white backdrop-blur-sm">
                    {timeRangeOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
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

      {/* Water Intake Analysis - Full Width */}
      <Card className="backdrop-blur-sm bg-white/95 border border-blue-100">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div className="flex items-center gap-2">
            <Droplets className="h-4 w-4 text-blue-600" />
            <CardTitle className="text-lg font-medium">Water Intake Analysis</CardTitle>
          </div>
          <div className="text-sm text-blue-900/70">
            {`${mockData.waterData.reduce((sum, data) => sum + data.amount, 0)}ml consumed`}
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={mockData.waterData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <defs>
                  <linearGradient id="waterGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0.2}/>
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
                  yAxisId="left"
                  domain={[0, 500]}
                  fontSize={11}
                  stroke="#64748b"
                  label={{ value: 'Water Intake (ml)', angle: -90, position: 'insideLeft', fontSize: 11 }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    borderRadius: '6px',
                    border: '1px solid rgba(14, 165, 233, 0.1)',
                    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)'
                  }}
                />
                <Bar 
                  yAxisId="left"
                  dataKey="amount" 
                  fill="url(#waterGradient)"
                  radius={[4, 4, 0, 0]}
                  barSize={30}
                />
                <Line 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="goal" 
                  stroke="#0284c7"
                  strokeWidth={2}
                  dot={false}
                  strokeDasharray="5 5"
                />
                <ReferenceLine 
                  yAxisId="left"
                  y={2000} 
                  stroke="#0ea5e9" 
                  strokeDasharray="3 3"
                  label={{ value: 'Daily Goal: 2000ml', position: 'right', fontSize: 11 }}
                />
                <Legend />
                <Brush 
                  dataKey="timestamp" 
                  height={30} 
                  stroke="#0ea5e9"
                  tickFormatter={formatXAxis}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-4 text-center">
            <div className="rounded-lg bg-blue-50 p-3">
              <p className="text-sm font-medium text-blue-900/70">Daily Goal</p>
              <p className="text-xl font-semibold text-blue-600">2000ml</p>
            </div>
            <div className="rounded-lg bg-blue-50 p-3">
              <p className="text-sm font-medium text-blue-900/70">Consumed</p>
              <p className="text-xl font-semibold text-blue-600">1150ml</p>
            </div>
            <div className="rounded-lg bg-blue-50 p-3">
              <p className="text-sm font-medium text-blue-900/70">Remaining</p>
              <p className="text-xl font-semibold text-blue-600">850ml</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
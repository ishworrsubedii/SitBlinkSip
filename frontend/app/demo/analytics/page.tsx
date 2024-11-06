"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Eye, Droplets, Timer, Activity } from "lucide-react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Legend 
} from 'recharts';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

// Mock data
const mockData = {
  weeklyStats: [
    { day: 'Mon', posture: 85, blinks: 15, water: 2000 },
    { day: 'Tue', posture: 78, blinks: 12, water: 1800 },
    { day: 'Wed', posture: 92, blinks: 14, water: 2200 },
    { day: 'Thu', posture: 88, blinks: 13, water: 1900 },
    { day: 'Fri', posture: 85, blinks: 15, water: 2100 },
    { day: 'Sat', posture: 90, blinks: 16, water: 1700 },
    { day: 'Sun', posture: 87, blinks: 14, water: 2300 },
  ]
};

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('week');

  return (
    <div className="w-full p-6 space-y-6">
      {/* Header with Time Range Selector */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-semibold text-gray-900">Analytics Overview</h1>
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select time range" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="day">Today</SelectItem>
            <SelectItem value="week">This Week</SelectItem>
            <SelectItem value="month">This Month</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Avg Posture Score</CardTitle>
            <Brain className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">87%</div>
            <p className="text-xs text-muted-foreground mt-1">+2.5% from last week</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Avg Blink Rate</CardTitle>
            <Eye className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">14 bpm</div>
            <p className="text-xs text-muted-foreground mt-1">Normal range</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Daily Water Intake</CardTitle>
            <Droplets className="h-4 w-4 text-cyan-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2.1L</div>
            <p className="text-xs text-muted-foreground mt-1">105% of goal</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Active Hours</CardTitle>
            <Timer className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">6.5h</div>
            <p className="text-xs text-muted-foreground mt-1">+1h from average</p>
          </CardContent>
        </Card>
      </div>

      {/* Weekly Trends Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-500" />
            Weekly Trends
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockData.weeklyStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-20" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="posture" stroke="#2563eb" name="Posture Score" />
                <Line type="monotone" dataKey="blinks" stroke="#16a34a" name="Blink Rate" />
                <Line type="monotone" dataKey="water" stroke="#0891b2" name="Water (ml)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
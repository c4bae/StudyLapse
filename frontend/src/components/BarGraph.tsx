"use client"

import { Bar, BarChart, CartesianGrid, XAxis } from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "../components/ui/chart"

export const description = "A bar chart"

const chartData = [
  { month: "January", desktop: 100 },
  { month: "February", desktop: 143 },
  { month: "March", desktop: 83 },
  { month: "April", desktop: 81 },
  { month: "May", desktop: 32 },
  { month: "June", desktop: 45 },
]

const chartConfig = {
  desktop: {
    label: "Hours",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig

export default function BarGraph({ fullName }) {
  return (
    <Card className="border-transparent bg-base-100 font-[DM_Sans]">
      <CardHeader>
        <CardTitle className="font-bold text-3xl text-neutral-700">Welcome back, {fullName}</CardTitle>
        <CardDescription>Here's how your study habits are showing up for this past 6 months:</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="w-full h-60">
          <BarChart accessibilityLayer data={chartData}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="month"
              tickLine={false}
              tickMargin={2}
              axisLine={false}
              tickFormatter={(value) => value.slice(0, 3)}
            />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel className="bg-white border-transparent"/>}
            />
            <Bar dataKey="desktop" fill="var(--color-desktop)" radius={8} barSize={80} />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}

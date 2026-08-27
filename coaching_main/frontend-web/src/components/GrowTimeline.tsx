/** GROW phase progression and distribution across the session. */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMemo } from "react";
import type { FinalMessage } from "../types";
import { Card, EmptyState } from "./ui/primitives";

const PHASES = ["Goal", "Reality", "Options", "Way Forward"] as const;

const PHASE_COLOR: Record<string, string> = {
  Goal: "#38bdf8",
  Reality: "#fbbf24",
  Options: "#4ade80",
  "Way Forward": "#c084fc",
};

export function GrowTimeline({ turns }: { turns: FinalMessage[] }) {
  const data = useMemo(() => {
    const counts = new Map<string, number>();
    for (const turn of turns) {
      const phase = turn.grow_phase.phase;
      counts.set(phase, (counts.get(phase) ?? 0) + 1);
    }
    const classified = PHASES.reduce((sum, p) => sum + (counts.get(p) ?? 0), 0);
    if (!classified) return [];
    // Percentages are a share of classified turns only, matching the
    // backend, so they always total 100%.
    return PHASES.filter((p) => counts.get(p)).map((phase) => ({
      phase,
      turns: counts.get(phase) ?? 0,
      percentage: ((counts.get(phase) ?? 0) / classified) * 100,
    }));
  }, [turns]);

  const sequence = useMemo(() => {
    const order: string[] = [];
    for (const turn of turns) {
      const phase = turn.grow_phase.phase;
      if (PHASES.includes(phase as (typeof PHASES)[number])) {
        if (order[order.length - 1] !== phase) order.push(phase);
      }
    }
    return order;
  }, [turns]);

  if (!data.length) {
    return (
      <Card title="GROW phases">
        <EmptyState>No phase data yet.</EmptyState>
      </Card>
    );
  }

  return (
    <Card
      title="GROW phases"
      subtitle={sequence.length ? sequence.join(" → ") : undefined}
    >
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis
              dataKey="phase"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              axisLine={{ stroke: "#334155" }}
              tickLine={false}
            />
            <YAxis
              unit="%"
              domain={[0, 100]}
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "#26344933" }}
              contentStyle={{
                background: "#1e293b",
                border: "1px solid #334155",
                borderRadius: 8,
                color: "#e2e8f0",
                fontSize: 12,
              }}
              formatter={(value: number, _name, entry) => [
                `${value.toFixed(1)}% (${entry.payload.turns} turns)`,
                "share",
              ]}
            />
            <Bar dataKey="percentage" radius={[4, 4, 0, 0]}>
              {data.map((row) => (
                <Cell key={row.phase} fill={PHASE_COLOR[row.phase] ?? "#64748b"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

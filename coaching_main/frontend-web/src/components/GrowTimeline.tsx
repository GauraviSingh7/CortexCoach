/**
 * Time spent in each GROW phase.
 *
 * Horizontal bars: the phase names are words, and words read better along
 * the axis than rotated under it. Percentages are a share of classified
 * turns only, matching the backend, so they always total 100%.
 */

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FinalMessage } from "../types";
import { Card, EmptyState } from "./ui/primitives";

const PHASES = ["Goal", "Reality", "Options", "Way Forward"] as const;

/** Muted, natural pigments - nothing here should shout. */
const PHASE_COLOR: Record<string, string> = {
  Goal: "#7b96ac",
  Reality: "#c0925e",
  Options: "#86a07c",
  "Way Forward": "#9a8298",
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
    return PHASES.filter((phase) => counts.get(phase)).map((phase) => ({
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
      <Card title="Phases of the session">
        <EmptyState>No phase has opened yet.</EmptyState>
      </Card>
    );
  }

  return (
    <Card
      title="Phases of the session"
      subtitle={sequence.length ? sequence.join("  →  ") : undefined}
    >
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 0, right: 12, left: 0, bottom: 0 }}
            barCategoryGap="28%"
          >
            <XAxis type="number" domain={[0, 100]} hide />
            <YAxis
              type="category"
              dataKey="phase"
              width={96}
              tick={{ fill: "#6e655b", fontSize: 13 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "#f1ede5" }}
              contentStyle={{
                background: "#fdfcfa",
                border: "1px solid #e5dfd4",
                borderRadius: 10,
                color: "#2e2a26",
                fontSize: 13,
                boxShadow: "0 1px 2px rgb(46 42 38 / 0.06)",
              }}
              formatter={(value: number, _name, entry) => [
                `${value.toFixed(0)}% · ${entry.payload.turns} turns`,
                "",
              ]}
            />
            <Bar dataKey="percentage" radius={[3, 3, 3, 3]} barSize={14}>
              {data.map((row) => (
                <Cell key={row.phase} fill={PHASE_COLOR[row.phase] ?? "#a09688"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

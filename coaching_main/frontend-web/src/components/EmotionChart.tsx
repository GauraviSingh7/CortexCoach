/**
 * Emotional journey over the session.
 *
 * Turns with no emotional signal are omitted rather than plotted as
 * neutral - the backend returns {} for those on purpose, and drawing a
 * flat neutral line was exactly the artefact the review flagged.
 */

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FinalMessage } from "../types";
import { Card, EmptyState, SourceBadge } from "./ui/primitives";
import { clockTime, dominantEmotion } from "../lib/format";

export function EmotionChart({ turns }: { turns: FinalMessage[] }) {
  const data = useMemo(
    () =>
      turns
        .map((turn, index) => {
          const emotion = dominantEmotion(turn.emotion_trend);
          if (!emotion) return null;
          return {
            index,
            time: clockTime(turn.timestamp),
            speaker: turn.speaker,
            emotion: emotion.label,
            confidence: emotion.score,
          };
        })
        .filter((point): point is NonNullable<typeof point> => point !== null),
    [turns],
  );

  const source = turns.length ? turns[turns.length - 1].sources.emotion : undefined;

  return (
    <Card
      title="Emotional journey"
      subtitle={`${data.length} of ${turns.length} turns carried an emotional signal`}
      actions={<SourceBadge source={source} />}
    >
      {data.length === 0 ? (
        <EmptyState>No emotional signal detected yet.</EmptyState>
      ) : (
        <div className="h-52">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                axisLine={{ stroke: "#334155" }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  background: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: 8,
                  color: "#e2e8f0",
                  fontSize: 12,
                }}
                formatter={(value: number, _name, entry) => [
                  `${entry.payload.emotion} (${value.toFixed(2)})`,
                  entry.payload.speaker,
                ]}
              />
              <Line
                type="monotone"
                dataKey="confidence"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={{ r: 3, fill: "#38bdf8" }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}

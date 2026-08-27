/** Top row of live session metrics, each labelled with its provenance. */

import { Card, Metric, SourceBadge } from "./ui/primitives";
import { metric, percent } from "../lib/format";
import type { LiveStats } from "../store/session";

export function StatsBanner({ stats }: { stats: LiveStats }) {
  const engagementTone =
    stats.engagement >= 0.6 ? "ok" : stats.engagement >= 0.4 ? "warn" : "bad";

  return (
    <Card
      title="Live signals"
      subtitle={`${stats.turnCount} turns · ${stats.coachTurns} coach / ${stats.coacheeTurns} coachee`}
      actions={
        <div className="flex flex-wrap items-center gap-1.5">
          <SourceBadge source={stats.sources.engagement} />
        </div>
      }
    >
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        <Metric
          label="GROW phase"
          value={stats.growPhase}
          hint={
            stats.latest?.grow_phase.inherited ? "continuing" : "newly opened"
          }
        />
        <Metric
          label="Engagement"
          value={metric(stats.engagement)}
          tone={engagementTone}
        />
        <Metric label="Learning style" value={stats.learningStyle} />
        <Metric
          label="Sarcasm"
          value={stats.sarcasmCount}
          hint="turns flagged"
          tone={stats.sarcasmCount > 0 ? "warn" : "neutral"}
        />
        <Metric
          label="Off-topic"
          value={stats.digressionCount}
          hint="moments"
          tone={stats.digressionCount > 0 ? "warn" : "neutral"}
        />
        <Metric
          label="Topic drift"
          value={percent(stats.digression)}
          hint="current turn"
        />
      </div>
    </Card>
  );
}

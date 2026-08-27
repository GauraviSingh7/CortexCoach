/** A quiet summary of where the session currently stands. */

import { Card, Metric, SourceBadge } from "./ui/primitives";
import { metric, percent } from "../lib/format";
import type { LiveStats } from "../store/session";

export function StatsBanner({ stats }: { stats: LiveStats }) {
  const engagementTone =
    stats.engagement >= 0.6
      ? "good"
      : stats.engagement >= 0.4
        ? "neutral"
        : "attention";

  return (
    <Card
      title="Where the session is"
      subtitle={`${stats.coachTurns} coach turns · ${stats.coacheeTurns} coachee turns`}
      actions={<SourceBadge source={stats.sources.engagement} />}
    >
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5">
        <Metric
          label="Phase"
          value={stats.growPhase}
          hint={stats.latest?.grow_phase.inherited ? "continuing" : "just opened"}
        />
        <Metric
          label="Engagement"
          value={metric(stats.engagement)}
          tone={engagementTone}
        />
        <Metric label="Learning style" value={stats.learningStyle} />
        <Metric
          label="Sarcasm noticed"
          value={stats.sarcasmCount}
          hint={stats.sarcasmCount === 1 ? "moment" : "moments"}
          tone={stats.sarcasmCount > 0 ? "attention" : "neutral"}
        />
        <Metric
          label="Off topic"
          value={stats.digressionCount}
          hint={`drift now ${percent(stats.digression)}`}
          tone={stats.digressionCount > 0 ? "attention" : "neutral"}
        />
      </div>
    </Card>
  );
}

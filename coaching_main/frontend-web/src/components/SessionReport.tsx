/** Final session report, including GROW coverage and signal provenance. */

import type { SessionReport as Report, Source } from "../types";
import { Badge, Card, EmptyState, Metric, SourceBadge } from "./ui/primitives";
import { metric, percent } from "../lib/format";
import { ModelStatusPanel } from "./ModelStatusPanel";
import type { ModelStatusPayload } from "../types";

/** The report carries {} when no status was recorded; narrow it safely. */
function asModelStatus(value: Report["model_status"]): ModelStatusPayload | null {
  return "models" in value ? (value as ModelStatusPayload) : null;
}

function ProvenanceLegend({ sources }: { sources: Record<string, Source> }) {
  const entries = Object.entries(sources ?? {});
  if (!entries.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-[--color-ink-dim]">
      <span>Signal provenance:</span>
      {entries.sort().map(([key, source]) => (
        <span key={key} className="flex items-center gap-1">
          <span>{key}</span>
          <SourceBadge source={source} />
        </span>
      ))}
    </div>
  );
}

export function SessionReport({ report }: { report: Report | null }) {
  if (!report) {
    return (
      <Card title="Session report">
        <EmptyState>Complete a session to generate a report.</EmptyState>
      </Card>
    );
  }

  const effectiveness = report.coaching_effectiveness ?? {};
  const coverage = report.grow_coverage as Report["grow_coverage"] & {
    coverage_pct?: number;
    phases_missing?: string[];
    classified_turns?: number;
    total_turns?: number;
  };
  const vak = report.learning_style_analysis ?? {};
  const sarcasm = report.sarcasm_summary ?? {};
  const digression = report.digression_summary ?? {};
  const phaseTotal = report.grow_phases.reduce((s, r) => s + r.percentage, 0);

  const download = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `coaching_report_${report.session_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-3">
      <Card
        title="Session report"
        subtitle={`${report.session_id.slice(0, 8)} · ${report.duration_minutes.toFixed(1)} minutes`}
        actions={
          <button
            type="button"
            onClick={download}
            className="rounded-lg border border-[--color-line] px-3 py-1.5 text-sm hover:bg-[--color-panel-soft]"
          >
            Download JSON
          </button>
        }
      >
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric label="Overall" value={metric(effectiveness.overall)} />
          <Metric label="Questioning" value={metric(effectiveness.questioning)} />
          <Metric label="Listening" value={metric(effectiveness.listening)} />
          <Metric
            label="Engagement"
            value={metric(effectiveness.engagement_management)}
          />
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2">
          {Object.entries(report.participants ?? {}).map(([role, stats]) => (
            <Metric
              key={role}
              label={`${role} turns`}
              value={stats.total_turns ?? 0}
              hint={`avg ${Math.round(stats.avg_words ?? 0)} words`}
            />
          ))}
        </div>
      </Card>

      <Card
        title="GROW phases"
        subtitle={`Shares of classified turns · total ${phaseTotal.toFixed(1)}%`}
      >
        {report.grow_phases.length === 0 ? (
          <EmptyState>No phase data.</EmptyState>
        ) : (
          <ul className="flex flex-col gap-2">
            {report.grow_phases.map((row) => (
              <li key={row.phase}>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span>{row.phase}</span>
                  <span className="tabular-nums text-[--color-ink-dim]">
                    {row.percentage.toFixed(1)}% · {row.turns} turns
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-[--color-panel-soft]">
                  <div
                    className="h-full rounded-full bg-sky-500"
                    style={{ width: `${row.percentage}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        )}

        {coverage?.coverage_pct !== undefined && (
          <div className="mt-3 grid grid-cols-2 gap-2">
            <Metric
              label="Phase coverage"
              value={`${coverage.coverage_pct.toFixed(0)}%`}
              hint={`${coverage.classified_turns}/${coverage.total_turns} turns classified`}
            />
            <Metric
              label="Phases reached"
              value={`${4 - (coverage.phases_missing?.length ?? 0)}/4`}
              hint={
                coverage.phases_missing?.length
                  ? `missing: ${coverage.phases_missing.join(", ")}`
                  : "all four reached"
              }
              tone={coverage.phases_missing?.length ? "warn" : "ok"}
            />
          </div>
        )}
      </Card>

      <div className="grid gap-3 lg:grid-cols-2">
        <Card title="Conversation signals">
          <div className="grid grid-cols-2 gap-2">
            <Metric
              label="Sarcasm"
              value={sarcasm.count_detected ?? 0}
              hint={`of ${sarcasm.total_evaluated ?? 0} turns`}
              tone={(sarcasm.count_detected ?? 0) > 0 ? "warn" : "neutral"}
            />
            <Metric
              label="Off-topic"
              value={digression.off_topic_moments ?? 0}
              hint={`of ${digression.total_evaluated ?? 0} turns`}
              tone={(digression.off_topic_moments ?? 0) > 0 ? "warn" : "neutral"}
            />
          </div>
          {(sarcasm.moments?.length ?? 0) > 0 && (
            <ul className="mt-3 flex flex-col gap-1.5">
              {sarcasm.moments!.map((moment, index) => (
                <li
                  key={index}
                  className="rounded-lg bg-[--color-panel-soft] px-3 py-2 text-xs"
                >
                  <div className="mb-1 flex items-center gap-1.5">
                    <Badge tone={moment.speaker === "coach" ? "coach" : "coachee"}>
                      {moment.speaker}
                    </Badge>
                    <Badge tone="warn">{moment.type}</Badge>
                    <span className="tabular-nums text-[--color-ink-dim]">
                      {moment.score.toFixed(2)}
                    </span>
                  </div>
                  <p className="text-[--color-ink-dim]">{moment.text}</p>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Learning style (VAK)">
          {Object.keys(vak).length === 0 ? (
            <EmptyState>Insufficient data.</EmptyState>
          ) : (
            <div className="grid grid-cols-3 gap-2">
              <Metric label="Visual" value={percent(vak.visual)} />
              <Metric label="Auditory" value={percent(vak.auditory)} />
              <Metric label="Kinesthetic" value={percent(vak.kinesthetic)} />
            </div>
          )}
        </Card>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Card title="Key insights">
          <ul className="flex list-disc flex-col gap-1.5 pl-4 text-sm leading-relaxed">
            {report.key_insights.map((insight, index) => (
              <li key={index}>{insight}</li>
            ))}
          </ul>
        </Card>
        <Card title="Recommendations">
          <ul className="flex list-disc flex-col gap-1.5 pl-4 text-sm leading-relaxed">
            {report.recommendations.map((rec, index) => (
              <li key={index}>{rec}</li>
            ))}
          </ul>
        </Card>
      </div>

      <Card title="Summary">
        <p className="text-sm leading-relaxed">{report.transcript_summary}</p>
        <div className="mt-3">
          <ProvenanceLegend sources={report.analysis_sources} />
        </div>
      </Card>

      <ModelStatusPanel status={asModelStatus(report.model_status)} />
    </div>
  );
}

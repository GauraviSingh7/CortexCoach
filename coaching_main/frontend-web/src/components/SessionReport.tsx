/**
 * The write-up.
 *
 * Set as a document: a readable measure, serif headings, and prose given
 * room. This is the artefact a coach would actually sit with afterwards,
 * so it should read like notes rather than a metrics export.
 */

import type { ModelStatusPayload, SessionReport as Report, Source } from "../types";
import { Badge, Card, Divider, EmptyState, Metric, SourceBadge } from "./ui/primitives";
import { metric, percent } from "../lib/format";
import { ModelStatusPanel } from "./ModelStatusPanel";

/** The report carries {} when no status was recorded; narrow it safely. */
function asModelStatus(value: Report["model_status"]): ModelStatusPayload | null {
  return "models" in value ? (value as ModelStatusPayload) : null;
}

const PHASE_COLOR: Record<string, string> = {
  Goal: "#7b96ac",
  Reality: "#c0925e",
  Options: "#86a07c",
  "Way Forward": "#9a8298",
};

function Provenance({ sources }: { sources: Record<string, Source> }) {
  const entries = Object.entries(sources ?? {});
  if (!entries.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[13px] text-ink-soft">
      <span>Where each reading came from:</span>
      {entries.sort().map(([key, source]) => (
        <span key={key} className="flex items-center gap-1.5">
          <span>{key.replace(/_/g, " ")}</span>
          <SourceBadge source={source} />
        </span>
      ))}
    </div>
  );
}

export function SessionReport({ report }: { report: Report | null }) {
  if (!report) {
    return (
      <Card title="Session notes">
        <EmptyState>
          Run a session and the write-up will appear here.
        </EmptyState>
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
    <div className="mx-auto flex max-w-[860px] flex-col gap-4 pb-8">
      <Card
        title="Session notes"
        subtitle={`${report.duration_minutes.toFixed(0)} minutes · ${report.session_id.slice(0, 8)}`}
        actions={
          <button
            type="button"
            onClick={download}
            className="rounded-lg border border-rule bg-card px-3 py-1.5 text-[13px] text-ink-soft transition-colors hover:bg-sink"
          >
            Download JSON
          </button>
        }
      >
        <p className="max-w-[62ch] text-[15px] leading-[1.75] text-ink">
          {report.transcript_summary}
        </p>

        <Divider />

        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
          <Metric label="Overall" value={metric(effectiveness.overall)} />
          <Metric label="Questioning" value={metric(effectiveness.questioning)} />
          <Metric label="Listening" value={metric(effectiveness.listening)} />
          <Metric
            label="Engagement"
            value={metric(effectiveness.engagement_management)}
          />
        </div>

        <div className="mt-2.5 grid grid-cols-2 gap-2.5">
          {Object.entries(report.participants ?? {}).map(([role, stats]) => (
            <Metric
              key={role}
              label={`${role === "coach" ? "Coach" : "Coachee"} turns`}
              value={stats.total_turns ?? 0}
              hint={`around ${Math.round(stats.avg_words ?? 0)} words each`}
            />
          ))}
        </div>
      </Card>

      <Card
        title="How the session moved"
        subtitle="Share of the turns spent in each phase"
      >
        {report.grow_phases.length === 0 ? (
          <EmptyState>No phase was established.</EmptyState>
        ) : (
          <ul className="flex flex-col gap-3.5">
            {report.grow_phases.map((row) => (
              <li key={row.phase}>
                <div className="mb-1.5 flex items-baseline justify-between text-[13px]">
                  <span className="text-ink">{row.phase}</span>
                  <span className="tnum text-ink-soft">
                    {row.percentage.toFixed(0)}% · {row.turns} turns
                  </span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-sink">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${row.percentage}%`,
                      backgroundColor: PHASE_COLOR[row.phase] ?? "#a09688",
                    }}
                  />
                </div>
              </li>
            ))}
          </ul>
        )}

        {coverage?.coverage_pct !== undefined && (
          <p className="mt-4 text-[13px] leading-relaxed text-ink-soft">
            {coverage.classified_turns} of {coverage.total_turns} turns sat
            within a recognised phase
            {coverage.phases_missing?.length
              ? `, and the session did not reach ${coverage.phases_missing.join(" or ")}.`
              : ", and all four phases were reached."}
          </p>
        )}
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="What stood out">
          <div className="grid grid-cols-2 gap-2.5">
            <Metric
              label="Sarcasm"
              value={sarcasm.count_detected ?? 0}
              hint={`of ${sarcasm.total_evaluated ?? 0} turns`}
              tone={(sarcasm.count_detected ?? 0) > 0 ? "attention" : "neutral"}
            />
            <Metric
              label="Off topic"
              value={digression.off_topic_moments ?? 0}
              hint={`of ${digression.total_evaluated ?? 0} turns`}
              tone={
                (digression.off_topic_moments ?? 0) > 0 ? "attention" : "neutral"
              }
            />
          </div>

          {(sarcasm.moments?.length ?? 0) > 0 && (
            <ul className="mt-4 flex flex-col gap-3">
              {sarcasm.moments!.map((moment, index) => (
                <li key={index} className="border-l-2 border-clay/40 pl-3">
                  <div className="mb-1 flex items-center gap-2">
                    <Badge
                      tone={moment.speaker === "coach" ? "coach" : "coachee"}
                    >
                      {moment.speaker}
                    </Badge>
                    <span className="text-[12px] text-ink-faint">
                      {moment.type.replace(/_/g, " ")}
                    </span>
                  </div>
                  <p className="text-[14px] leading-relaxed text-ink-soft italic">
                    “{moment.text}”
                  </p>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="How they take things in">
          {Object.keys(vak).length === 0 ? (
            <EmptyState>Not enough to say.</EmptyState>
          ) : (
            <div className="grid grid-cols-3 gap-2.5">
              <Metric label="Visual" value={percent(vak.visual)} />
              <Metric label="Auditory" value={percent(vak.auditory)} />
              <Metric label="Kinesthetic" value={percent(vak.kinesthetic)} />
            </div>
          )}
        </Card>
      </div>

      <Card title="What we noticed">
        <ul className="flex flex-col gap-2.5">
          {report.key_insights.map((insight, index) => (
            <li
              key={index}
              className="max-w-[62ch] border-l-2 border-rule pl-3 text-[14px] leading-relaxed text-ink"
            >
              {insight}
            </li>
          ))}
        </ul>
      </Card>

      <Card title="Worth trying next time">
        <ul className="flex flex-col gap-2.5">
          {report.recommendations.map((rec, index) => (
            <li
              key={index}
              className="max-w-[62ch] border-l-2 border-sage/40 pl-3 text-[14px] leading-relaxed text-ink"
            >
              {rec}
            </li>
          ))}
        </ul>
        <Divider />
        <Provenance sources={report.analysis_sources} />
      </Card>

      <ModelStatusPanel status={asModelStatus(report.model_status)} />
    </div>
  );
}

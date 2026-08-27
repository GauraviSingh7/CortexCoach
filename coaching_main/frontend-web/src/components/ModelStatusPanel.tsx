/**
 * What produced these numbers.
 *
 * The Streamlit dashboard had no model diagnostics at all, so a session
 * with three models silently on fallback looked identical to a healthy
 * one. This panel makes that plain, in ordinary language, without turning
 * it into an alarm - degraded is a known state here, not a failure.
 */

import { useState } from "react";
import type { ModelState, ModelStatusPayload } from "../types";
import { Badge, Card, EmptyState } from "./ui/primitives";

const STATE_TONE: Record<ModelState, "good" | "attention" | "neutral"> = {
  trained: "good",
  heuristic: "attention",
  unavailable: "neutral",
};

const STATE_LABEL: Record<ModelState, string> = {
  trained: "trained model",
  heuristic: "estimated",
  unavailable: "unavailable",
};

/** "vak_inference" reads better as "Vak inference" in a sentence. */
const humanize = (name: string) =>
  name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

export function ModelStatusPanel({
  status,
}: {
  status: ModelStatusPayload | null;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!status) {
    return (
      <Card title="Where the numbers come from">
        <EmptyState>Can't reach the backend to check.</EmptyState>
      </Card>
    );
  }

  const degraded = status.degraded ?? [];

  return (
    <Card
      title="Where the numbers come from"
      subtitle={
        degraded.length
          ? `${degraded.length} of ${status.total_count} signals are estimated`
          : `All ${status.total_count} models are running trained weights`
      }
    >
      {degraded.length > 0 && (
        <p className="mb-4 rounded-lg border border-clay/20 bg-clay-soft px-3.5 py-3 text-[13px] leading-relaxed text-ink">
          Some trained models can't be loaded, so those signals come from
          documented rule-based estimates instead. They're marked
          <span className="mx-1 inline-flex items-center rounded-md border border-clay/20 bg-clay-soft px-1.5 text-[12px] text-clay">
            estimated
          </span>
          wherever they appear.
        </p>
      )}

      <ul className="flex flex-col divide-y divide-rule-soft">
        {Object.entries(status.models).map(([name, model]) => {
          const open = expanded === name;
          return (
            <li key={name}>
              <button
                type="button"
                onClick={() => setExpanded(open ? null : name)}
                className="flex w-full items-center justify-between gap-3 py-2.5 text-left"
              >
                <span className="text-[14px] text-ink">{humanize(name)}</span>
                <span className="flex shrink-0 items-center gap-2">
                  <Badge tone={STATE_TONE[model.state]}>
                    {STATE_LABEL[model.state]}
                  </Badge>
                  <span className="text-[13px] text-ink-faint">
                    {open ? "−" : "+"}
                  </span>
                </span>
              </button>

              {open && (
                <div className="pb-3.5 text-[13px] leading-relaxed text-ink-soft">
                  <p>{model.detail}</p>
                  {model.weights_loaded && (
                    <p className="mt-1.5">
                      Weights:{" "}
                      <code className="rounded bg-sink px-1 py-0.5 text-[12px]">
                        {model.weights_loaded}
                      </code>
                    </p>
                  )}
                  {model.blocking_reason && (
                    <p className="mt-2 border-l-2 border-rule pl-3 text-ink">
                      {model.blocking_reason}
                    </p>
                  )}
                  {model.artifacts_missing.length > 0 && (
                    <p className="mt-1.5 text-ink-faint">
                      Missing: {model.artifacts_missing.join(", ")}
                    </p>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </Card>
  );
}

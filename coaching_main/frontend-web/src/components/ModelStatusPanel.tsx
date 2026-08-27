/**
 * Which models are genuinely running, and why the others are not.
 *
 * The Streamlit dashboard had no model diagnostics at all, so a session
 * with three models silently on fallback looked identical to a healthy
 * one. This panel makes that impossible to miss.
 */

import { useState } from "react";
import type { ModelStatusPayload, ModelState } from "../types";
import { Badge, Card, EmptyState } from "./ui/primitives";

const STATE_TONE: Record<ModelState, "ok" | "warn" | "bad"> = {
  trained: "ok",
  heuristic: "warn",
  unavailable: "bad",
};

const STATE_LABEL: Record<ModelState, string> = {
  trained: "Trained model",
  heuristic: "Rule-based heuristic",
  unavailable: "Not available",
};

export function ModelStatusPanel({
  status,
}: {
  status: ModelStatusPayload | null;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!status) {
    return (
      <Card title="Model status">
        <EmptyState>Model status unavailable — is the backend running?</EmptyState>
      </Card>
    );
  }

  const degraded = status.degraded ?? [];

  return (
    <Card
      title="Model status"
      subtitle={`${status.trained_count}/${status.total_count} using trained weights`}
    >
      {degraded.length > 0 && (
        <p className="mb-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-200">
          <strong>{degraded.length} model(s) are not using trained weights:</strong>{" "}
          {degraded.join(", ")}. Metrics from these come from documented
          rule-based heuristics.
        </p>
      )}

      <ul className="flex flex-col gap-1.5">
        {Object.entries(status.models).map(([name, model]) => {
          const open = expanded === name;
          return (
            <li key={name} className="rounded-lg bg-[--color-panel-soft]">
              <button
                type="button"
                onClick={() => setExpanded(open ? null : name)}
                className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
              >
                <span className="text-sm">{name}</span>
                <span className="flex items-center gap-2">
                  <Badge tone={STATE_TONE[model.state]}>
                    {STATE_LABEL[model.state]}
                  </Badge>
                  <span className="text-[--color-ink-dim]">{open ? "−" : "+"}</span>
                </span>
              </button>

              {open && (
                <div className="border-t border-[--color-line] px-3 py-2 text-xs leading-relaxed text-[--color-ink-dim]">
                  <p>{model.detail}</p>
                  {model.weights_loaded && (
                    <p className="mt-1">
                      Weights: <code>{model.weights_loaded}</code>
                    </p>
                  )}
                  {model.blocking_reason && (
                    <p className="mt-2 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1.5 text-rose-200">
                      <strong>Why it is not running:</strong>{" "}
                      {model.blocking_reason}
                    </p>
                  )}
                  {model.artifacts_missing.length > 0 && (
                    <p className="mt-1">
                      Missing: {model.artifacts_missing.join(", ")}
                    </p>
                  )}
                  {model.artifacts_found.length > 0 && (
                    <p className="mt-1">
                      Present: {model.artifacts_found.join(", ")}
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

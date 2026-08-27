/** Application header with an at-a-glance model health indicator. */

import type { ModelStatusPayload } from "../types";
import { Badge } from "./ui/primitives";

export function Header({ status }: { status: ModelStatusPayload | null }) {
  const degraded = status?.degraded?.length ?? 0;

  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-[--color-line] px-5 py-3">
      <div>
        <h1 className="text-base font-semibold">AI Coaching Observer</h1>
        <p className="text-xs text-[--color-ink-dim]">
          Real-time GROW, engagement and conversation-signal analysis
        </p>
      </div>

      {status && (
        <Badge tone={degraded ? "warn" : "ok"}>
          {status.trained_count}/{status.total_count} models trained
          {degraded ? ` · ${degraded} on heuristics` : ""}
        </Badge>
      )}
    </header>
  );
}

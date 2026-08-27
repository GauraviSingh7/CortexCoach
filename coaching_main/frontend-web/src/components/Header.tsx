/** Masthead. Model health sits here so it is seen before any number is. */

import type { ModelStatusPayload } from "../types";
import { Badge } from "./ui/primitives";

export function Header({ status }: { status: ModelStatusPayload | null }) {
  const degraded = status?.degraded?.length ?? 0;

  return (
    <header className="border-b border-rule bg-card px-7 py-5">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-baseline justify-between gap-3">
        <div>
          <h1 className="font-serif text-[22px] leading-tight text-ink">
            Coaching Observer
          </h1>
          <p className="mt-0.5 text-[13px] text-ink-soft">
            Notes on a conversation — phases, engagement and what was heard
          </p>
        </div>

        {status && (
          <Badge
            tone={degraded ? "attention" : "good"}
            title={
              degraded
                ? `${degraded} model(s) running on rule-based heuristics`
                : "All models running trained weights"
            }
          >
            {degraded
              ? `${degraded} of ${status.total_count} signals estimated`
              : `${status.total_count} models trained`}
          </Badge>
        )}
      </div>
    </header>
  );
}

/**
 * Small presentational primitives.
 *
 * Hand-written rather than pulled in via the shadcn/ui generator: the
 * dashboard needs six components, and vendoring a component library plus
 * its Radix dependency tree for that would be more surface than value.
 */

import type { ReactNode } from "react";
import type { Source } from "../../types";

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = "",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-[--color-line] bg-[--color-panel] p-4 ${className}`}
    >
      {(title || actions) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && (
              <h2 className="text-sm font-semibold tracking-wide text-[--color-ink]">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-0.5 text-xs text-[--color-ink-dim]">{subtitle}</p>
            )}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Metric({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "neutral" | "ok" | "warn" | "bad";
}) {
  const toneClass = {
    neutral: "text-[--color-ink]",
    ok: "text-[--color-ok]",
    warn: "text-[--color-warn]",
    bad: "text-[--color-bad]",
  }[tone];

  return (
    <div className="rounded-lg bg-[--color-panel-soft] px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-wider text-[--color-ink-dim]">
        {label}
      </div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${toneClass}`}>
        {value}
      </div>
      {hint && (
        <div className="mt-0.5 text-[11px] text-[--color-ink-dim]">{hint}</div>
      )}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "ok" | "warn" | "bad" | "coach" | "coachee";
  title?: string;
}) {
  const toneClass = {
    neutral: "bg-slate-700/60 text-slate-200",
    ok: "bg-emerald-500/15 text-emerald-300",
    warn: "bg-amber-500/15 text-amber-300",
    bad: "bg-rose-500/15 text-rose-300",
    coach: "bg-sky-500/15 text-sky-300",
    coachee: "bg-purple-500/15 text-purple-300",
  }[tone];

  return (
    <span
      title={title}
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${toneClass}`}
    >
      {children}
    </span>
  );
}

/**
 * Labels whether a number came from a trained model or a heuristic.
 * Every degraded signal in the app carries one of these.
 */
export function SourceBadge({ source }: { source?: Source }) {
  if (!source) return null;
  const config = {
    model: { tone: "ok" as const, label: "model", title: "From a trained model" },
    heuristic: {
      tone: "warn" as const,
      label: "heuristic",
      title: "Rule-based estimate, not a trained model",
    },
    unavailable: {
      tone: "bad" as const,
      label: "no signal",
      title: "No value could be produced for this turn",
    },
  }[source];
  return (
    <Badge tone={config.tone} title={config.title}>
      {config.label}
    </Badge>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "danger" | "ghost";
}) {
  const variantClass = {
    primary: "bg-sky-600 hover:bg-sky-500 text-white",
    danger: "bg-rose-600 hover:bg-rose-500 text-white",
    ghost:
      "bg-transparent hover:bg-[--color-panel-soft] text-[--color-ink] border border-[--color-line]",
  }[variant];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${variantClass}`}
    >
      {children}
    </button>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-[--color-line] px-4 py-8 text-center text-sm text-[--color-ink-dim]">
      {children}
    </div>
  );
}

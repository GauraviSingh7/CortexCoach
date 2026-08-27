/**
 * Presentational primitives.
 *
 * Hand-written rather than pulled in via the shadcn/ui generator: the
 * dashboard needs six components, and vendoring a component library plus
 * its Radix dependency tree for that would be more surface than value.
 *
 * Two conventions run through all of them:
 *   - labels are sentence case, never uppercase-tracked;
 *   - colour is used to mean something, not to decorate.
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
      className={`lift rounded-xl border border-rule bg-card p-5 ${className}`}
    >
      {(title || actions) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div className="min-w-0">
            {title && <h2 className="text-[15px] text-ink">{title}</h2>}
            {subtitle && (
              <p className="mt-0.5 text-[13px] leading-snug text-ink-soft">
                {subtitle}
              </p>
            )}
          </div>
          {actions && <div className="shrink-0">{actions}</div>}
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
  tone?: "neutral" | "good" | "attention" | "alert";
}) {
  const toneClass = {
    neutral: "text-ink",
    good: "text-sage",
    attention: "text-clay",
    alert: "text-brick",
  }[tone];

  return (
    <div className="rounded-lg border border-rule-soft bg-sink px-3.5 py-3">
      <div className="text-[12px] leading-none text-ink-soft">{label}</div>
      <div className={`tnum mt-1.5 font-serif text-[22px] leading-tight ${toneClass}`}>
        {value}
      </div>
      {hint && (
        <div className="mt-1 text-[12px] leading-snug text-ink-faint">{hint}</div>
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
  tone?: "neutral" | "good" | "attention" | "alert" | "coach" | "coachee";
  title?: string;
}) {
  const toneClass = {
    neutral: "bg-sink text-ink-soft border-rule",
    good: "bg-sage-soft text-sage border-sage/20",
    attention: "bg-clay-soft text-clay border-clay/20",
    alert: "bg-brick-soft text-brick border-brick/20",
    coach: "bg-coach-soft text-coach border-coach/20",
    coachee: "bg-coachee-soft text-coachee border-coachee/20",
  }[tone];

  return (
    <span
      title={title}
      className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[12px] leading-5 ${toneClass}`}
    >
      {children}
    </span>
  );
}

/**
 * Says whether a number came from a trained model or a documented
 * heuristic. Every degraded signal in the app carries one.
 */
export function SourceBadge({ source }: { source?: Source }) {
  if (!source) return null;
  const config = {
    model: {
      tone: "good" as const,
      label: "trained model",
      title: "Produced by a trained model",
    },
    heuristic: {
      tone: "attention" as const,
      label: "estimated",
      title: "Rule-based estimate, not a trained model",
    },
    unavailable: {
      tone: "neutral" as const,
      label: "no reading",
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
  variant?: "primary" | "quiet" | "danger";
}) {
  const variantClass = {
    primary: "bg-sage text-white hover:bg-sage/90 border-transparent",
    quiet: "bg-card text-ink hover:bg-sink border-rule",
    danger: "bg-card text-brick hover:bg-brick-soft border-brick/30",
  }[variant];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg border px-3.5 py-2 text-[14px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${variantClass}`}
    >
      {children}
    </button>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-lg border border-dashed border-rule px-4 py-8 text-center text-[14px] text-ink-faint">
      {children}
    </p>
  );
}

/** A quiet horizontal rule for separating sections inside a card. */
export function Divider() {
  return <hr className="my-4 border-0 border-t border-rule-soft" />;
}

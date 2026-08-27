/**
 * How the coachee's feeling moved through the session.
 *
 * Drawn as a labelled arc rather than a line chart. What matters here is
 * the *sequence* of named feelings - stuck, then exposed, then relieved -
 * not a confidence value plotted against time, which was never a
 * meaningful quantity to graph.
 *
 * Turns with no emotional reading are omitted rather than plotted as
 * neutral: the backend returns nothing for those on purpose, and drawing
 * a flat neutral line was exactly the artefact the review flagged.
 */

import { useMemo } from "react";
import type { FinalMessage } from "../types";
import { Card, EmptyState, SourceBadge } from "./ui/primitives";
import { clockTime, dominantEmotion } from "../lib/format";

/**
 * Feelings grouped by how they sit, so the arc reads at a glance without
 * needing a legend. Unknown labels fall back to neutral.
 */
const TONE: Record<string, string> = {
  happy: "bg-sage-soft text-sage border-sage/25",
  excited: "bg-sage-soft text-sage border-sage/25",
  hopeful: "bg-sage-soft text-sage border-sage/25",
  relieved: "bg-sage-soft text-sage border-sage/25",
  sad: "bg-coach-soft text-coach border-coach/25",
  anxious: "bg-clay-soft text-clay border-clay/25",
  conflicted: "bg-clay-soft text-clay border-clay/25",
  frustrated: "bg-brick-soft text-brick border-brick/25",
};

const NEUTRAL_TONE = "bg-sink text-ink-soft border-rule";

export function EmotionChart({ turns }: { turns: FinalMessage[] }) {
  const points = useMemo(
    () =>
      turns
        .map((turn) => {
          const emotion = dominantEmotion(turn.emotion_trend);
          if (!emotion) return null;
          return {
            speaker: turn.speaker,
            label: emotion.label,
            score: emotion.score,
            time: clockTime(turn.timestamp),
          };
        })
        .filter((point): point is NonNullable<typeof point> => point !== null),
    [turns],
  );

  const source = turns.length
    ? turns[turns.length - 1].sources.emotion
    : undefined;

  return (
    <Card
      title="How it felt"
      subtitle={
        points.length
          ? `${points.length} of ${turns.length} turns carried a feeling`
          : undefined
      }
      actions={<SourceBadge source={source} />}
    >
      {points.length === 0 ? (
        <EmptyState>Nothing read as emotionally marked yet.</EmptyState>
      ) : (
        <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-2.5">
          {points.map((point, index) => (
            <li key={index} className="flex items-center gap-1.5">
              <span
                title={`${point.speaker} · ${point.time} · confidence ${point.score.toFixed(2)}`}
                className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[12px] leading-5 ${
                  TONE[point.label] ?? NEUTRAL_TONE
                }`}
              >
                {point.label}
              </span>
              {index < points.length - 1 && (
                <span aria-hidden className="text-[12px] text-ink-faint">
                  →
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}

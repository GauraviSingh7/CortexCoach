/**
 * Growing transcript of completed turns, plus any in-progress utterance.
 *
 * Turns are appended, never re-rendered wholesale, and the panel only
 * auto-scrolls when the reader is already at the bottom - so reading back
 * through earlier turns is not yanked away by incoming ones. That is the
 * behaviour the Streamlit version could not offer at all, because every
 * update re-ran the whole script and rebuilt the list from scratch.
 */

import { memo, useEffect, useRef } from "react";
import type { FinalMessage, Speaker } from "../types";
import { Badge, Card, EmptyState, SourceBadge } from "./ui/primitives";
import { clockTime, dominantEmotion, percent, titleCase } from "../lib/format";

const Turn = memo(function Turn({ turn }: { turn: FinalMessage }) {
  const isCoach = turn.speaker === "coach";
  const emotion = dominantEmotion(turn.emotion_trend);

  return (
    <li
      className="rounded-lg border-l-4 bg-[--color-panel-soft] px-3 py-2.5"
      style={{
        borderLeftColor: isCoach ? "var(--color-coach)" : "var(--color-coachee)",
      }}
    >
      <div className="mb-1 flex flex-wrap items-center gap-1.5">
        <Badge tone={isCoach ? "coach" : "coachee"}>
          {titleCase(turn.speaker)}
        </Badge>
        <span className="text-[11px] text-[--color-ink-dim]">
          {clockTime(turn.timestamp)}
        </span>
        <Badge>{turn.grow_phase.phase}</Badge>
        {emotion ? (
          <Badge title={`confidence ${percent(emotion.score)}`}>
            {emotion.label}
          </Badge>
        ) : (
          <Badge title="No emotional signal detected in this turn">
            no emotion signal
          </Badge>
        )}
        {turn.sarcasm_detected && (
          <Badge tone="warn" title={`score ${turn.sarcasm_score.toFixed(2)}`}>
            sarcasm: {turn.sarcasm_type}
          </Badge>
        )}
        {turn.digression_detected && <Badge tone="warn">off-topic</Badge>}
      </div>

      <p className="text-sm leading-relaxed text-[--color-ink]">
        {turn.transcript}
      </p>

      <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[11px] text-[--color-ink-dim]">
        <span>engagement {turn.engagement_score.toFixed(2)}</span>
        <SourceBadge source={turn.sources.emotion} />
      </div>
    </li>
  );
});

function Partial({ speaker, text }: { speaker: Speaker; text: string }) {
  if (!text) return null;
  const isCoach = speaker === "coach";
  return (
    <li
      className="rounded-lg border-l-4 border-dashed bg-[--color-panel-soft]/50 px-3 py-2.5 opacity-70"
      style={{
        borderLeftColor: isCoach ? "var(--color-coach)" : "var(--color-coachee)",
      }}
    >
      <Badge tone={isCoach ? "coach" : "coachee"}>{titleCase(speaker)}</Badge>
      <p className="mt-1 text-sm italic text-[--color-ink-dim]">{text}…</p>
    </li>
  );
}

export function LiveTranscript({
  turns,
  partials,
}: {
  turns: FinalMessage[];
  partials: Record<Speaker, string>;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedToBottom = useRef(true);

  useEffect(() => {
    const node = scrollRef.current;
    if (node && pinnedToBottom.current) {
      node.scrollTop = node.scrollHeight;
    }
  }, [turns.length, partials.coach, partials.coachee]);

  const onScroll = () => {
    const node = scrollRef.current;
    if (!node) return;
    const distanceFromBottom =
      node.scrollHeight - node.scrollTop - node.clientHeight;
    pinnedToBottom.current = distanceFromBottom < 48;
  };

  return (
    <Card
      title="Transcript"
      subtitle={`${turns.length} completed turns`}
      className="flex h-full min-h-0 flex-col"
    >
      {turns.length === 0 && !partials.coach && !partials.coachee ? (
        <EmptyState>Waiting for the first turn…</EmptyState>
      ) : (
        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="scroll-panel min-h-0 flex-1"
        >
          <ul className="flex flex-col gap-2 pr-1">
            {turns.map((turn, index) => (
              <Turn key={`${turn.timestamp}-${index}`} turn={turn} />
            ))}
            <Partial speaker="coach" text={partials.coach} />
            <Partial speaker="coachee" text={partials.coachee} />
          </ul>
        </div>
      )}
    </Card>
  );
}

/**
 * The conversation as it unfolds.
 *
 * This is the centre of the page and is set to be read, not scanned:
 * generous line height, a comfortable measure, and the speaker carried by
 * a quiet rule in the margin rather than a coloured block. Analysis sits
 * underneath each turn in small type, so it annotates the conversation
 * instead of competing with it.
 *
 * Turns are appended, never re-rendered wholesale, and the panel only
 * auto-scrolls when the reader is already at the bottom - so reading back
 * through earlier turns is not yanked away by incoming ones. The
 * Streamlit version could not do this at all: every update re-ran the
 * whole script and rebuilt the list from scratch.
 */

import { memo, useEffect, useRef } from "react";
import type { FinalMessage, Speaker } from "../types";
import { Badge, Card, EmptyState } from "./ui/primitives";
import { clockTime, dominantEmotion, titleCase } from "../lib/format";

const SPEAKER_RULE: Record<Speaker, string> = {
  coach: "border-coach/45",
  coachee: "border-coachee/45",
};

const SPEAKER_NAME: Record<Speaker, string> = {
  coach: "text-coach",
  coachee: "text-coachee",
};

const Turn = memo(function Turn({ turn }: { turn: FinalMessage }) {
  const emotion = dominantEmotion(turn.emotion_trend);

  return (
    <li className={`border-l-2 pl-4 ${SPEAKER_RULE[turn.speaker]}`}>
      <div className="flex items-baseline gap-2">
        <span
          className={`text-[13px] font-medium ${SPEAKER_NAME[turn.speaker]}`}
        >
          {titleCase(turn.speaker)}
        </span>
        <span className="tnum text-[12px] text-ink-faint">
          {clockTime(turn.timestamp)}
        </span>
      </div>

      <p className="mt-1 max-w-[62ch] text-[15px] leading-[1.7] text-ink">
        {turn.transcript}
      </p>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <Badge
          title={
            turn.grow_phase.inherited
              ? `Continuing the ${turn.grow_phase.phase} phase`
              : turn.grow_phase.reasoning
          }
        >
          {turn.grow_phase.phase}
        </Badge>

        {emotion ? (
          <Badge title={`confidence ${emotion.score.toFixed(2)}`}>
            {emotion.label}
          </Badge>
        ) : (
          <span className="text-[12px] text-ink-faint">no emotional reading</span>
        )}

        {turn.sarcasm_detected && (
          <Badge
            tone="attention"
            title={`score ${turn.sarcasm_score.toFixed(2)}`}
          >
            {turn.sarcasm_type.replace(/_/g, " ")}
          </Badge>
        )}

        {turn.digression_detected && <Badge tone="attention">off topic</Badge>}
      </div>
    </li>
  );
});

function Partial({ speaker, text }: { speaker: Speaker; text: string }) {
  if (!text) return null;
  return (
    <li className={`border-l-2 border-dotted pl-4 ${SPEAKER_RULE[speaker]}`}>
      <span className={`text-[13px] font-medium ${SPEAKER_NAME[speaker]}`}>
        {titleCase(speaker)}
      </span>
      <p className="mt-1 max-w-[62ch] text-[15px] leading-[1.7] italic text-ink-faint">
        {text}…
      </p>
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
    pinnedToBottom.current =
      node.scrollHeight - node.scrollTop - node.clientHeight < 48;
  };

  return (
    <Card
      title="The conversation"
      subtitle={`${turns.length} turns so far`}
      className="flex h-full min-h-0 flex-col"
    >
      {turns.length === 0 && !partials.coach && !partials.coachee ? (
        <EmptyState>Waiting for the first turn…</EmptyState>
      ) : (
        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="scroll-panel -mr-2 min-h-0 flex-1 pr-2"
        >
          <ul className="flex flex-col gap-6">
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

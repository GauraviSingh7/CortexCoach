/** Starting and ending a session. */

import { useRef, useState } from "react";
import { Badge, Button, Card, Divider } from "./ui/primitives";
import type { ConnectionState } from "../lib/ws";

export function ControlPanel({
  active,
  busy,
  sessionId,
  connection,
  onStartLive,
  onStartFile,
  onStartReplay,
  onStop,
}: {
  active: boolean;
  busy: boolean;
  sessionId: string | null;
  connection: ConnectionState;
  onStartLive: (coachSpeakerId: string | null) => void;
  onStartFile: (file: File, coachSpeakerId: string | null) => void;
  onStartReplay: () => void;
  onStop: () => void;
}) {
  const [coachSpeaker, setCoachSpeaker] = useState<string>("");
  const fileInput = useRef<HTMLInputElement>(null);

  const pinned = coachSpeaker.trim() ? coachSpeaker.trim().toUpperCase() : null;

  const connectionLabel =
    connection === "open"
      ? "listening"
      : connection === "connecting"
        ? "connecting"
        : "not listening";

  return (
    <Card
      title="Session"
      subtitle={sessionId ? `${sessionId.slice(0, 8)}` : "Nothing running"}
      actions={
        <Badge tone={connection === "open" ? "good" : "neutral"}>
          {connectionLabel}
        </Badge>
      }
    >
      <div className="flex flex-col gap-4">
        <label className="flex flex-col gap-1.5 text-[13px] text-ink-soft">
          Which speaker is the coach?
          <input
            value={coachSpeaker}
            onChange={(event) => setCoachSpeaker(event.target.value)}
            placeholder="A or B — or leave blank to work it out"
            disabled={active}
            className="rounded-lg border border-rule bg-card px-3 py-2 text-[14px] text-ink outline-none placeholder:text-ink-faint focus:border-sage/50 disabled:opacity-50"
          />
        </label>

        <div className="flex flex-wrap gap-2">
          <Button onClick={() => onStartLive(pinned)} disabled={active || busy}>
            Start listening
          </Button>
          <Button
            variant="quiet"
            onClick={() => fileInput.current?.click()}
            disabled={active || busy}
          >
            Upload a recording
          </Button>
          <Button variant="danger" onClick={onStop} disabled={!active || busy}>
            {busy ? "Working…" : "End session"}
          </Button>
        </div>

        <Divider />

        <div>
          <p className="text-[13px] leading-relaxed text-ink-soft">
            No API keys to hand? Replay a stored 40-turn session through the
            same analysis.
          </p>
          <div className="mt-2.5">
            <Button
              variant="quiet"
              onClick={onStartReplay}
              disabled={active || busy}
            >
              Replay a sample session
            </Button>
          </div>
        </div>

        <input
          ref={fileInput}
          type="file"
          accept="audio/*,.wav,.mp3,.m4a"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onStartFile(file, pinned);
            event.target.value = "";
          }}
        />
      </div>
    </Card>
  );
}

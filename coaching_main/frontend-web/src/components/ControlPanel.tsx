/** Session controls: start live or from a file, and stop. */

import { useRef, useState } from "react";
import { Badge, Button, Card } from "./ui/primitives";
import type { ConnectionState } from "../lib/ws";

export function ControlPanel({
  active,
  busy,
  sessionId,
  connection,
  onStartLive,
  onStartFile,
  onStop,
}: {
  active: boolean;
  busy: boolean;
  sessionId: string | null;
  connection: ConnectionState;
  onStartLive: (coachSpeakerId: string | null) => void;
  onStartFile: (file: File, coachSpeakerId: string | null) => void;
  onStop: () => void;
}) {
  const [coachSpeaker, setCoachSpeaker] = useState<string>("");
  const fileInput = useRef<HTMLInputElement>(null);

  const pinned = coachSpeaker.trim() ? coachSpeaker.trim().toUpperCase() : null;

  const connectionTone =
    connection === "open" ? "ok" : connection === "connecting" ? "warn" : "bad";

  return (
    <Card
      title="Session"
      subtitle={sessionId ? `id ${sessionId.slice(0, 8)}` : "No active session"}
      actions={
        <Badge tone={connectionTone}>
          {connection === "open"
            ? "live"
            : connection === "connecting"
              ? "connecting"
              : "offline"}
        </Badge>
      }
    >
      <div className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-xs text-[--color-ink-dim]">
          Coach speaker label (optional)
          <input
            value={coachSpeaker}
            onChange={(event) => setCoachSpeaker(event.target.value)}
            placeholder="A or B — leave blank to detect automatically"
            disabled={active}
            className="rounded-lg border border-[--color-line] bg-[--color-panel-soft] px-2.5 py-1.5 text-sm text-[--color-ink] placeholder:text-slate-500 disabled:opacity-50"
          />
        </label>

        <div className="flex flex-wrap gap-2">
          <Button onClick={() => onStartLive(pinned)} disabled={active || busy}>
            Start live session
          </Button>
          <Button
            variant="ghost"
            onClick={() => fileInput.current?.click()}
            disabled={active || busy}
          >
            Upload audio file
          </Button>
          <Button variant="danger" onClick={onStop} disabled={!active || busy}>
            {busy ? "Working…" : "Stop session"}
          </Button>
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

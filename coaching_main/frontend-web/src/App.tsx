/**
 * Dashboard shell.
 *
 * Owns the session reducer, opens the WebSocket while a session is live,
 * and lays out the live view or the final report. Every panel below reads
 * from the same reducer, so an incoming turn re-renders only the pieces
 * bound to what actually changed.
 */

import { useCallback, useEffect } from "react";
import * as api from "./lib/api";
import { useFeedbackSocket } from "./lib/ws";
import { useLiveStats, useSession } from "./store/session";
import { ControlPanel } from "./components/ControlPanel";
import { EmotionChart } from "./components/EmotionChart";
import { GrowTimeline } from "./components/GrowTimeline";
import { Header } from "./components/Header";
import { LiveTranscript } from "./components/LiveTranscript";
import { ModelStatusPanel } from "./components/ModelStatusPanel";
import { SessionReport } from "./components/SessionReport";
import { StatsBanner } from "./components/StatsBanner";
import { Suggestions } from "./components/Suggestions";

export default function App() {
  const { state, dispatch, onFeedback } = useSession();
  const connection = useFeedbackSocket(state.active, onFeedback);
  const stats = useLiveStats(state.turns);

  // Model status is fetched once at mount and refreshed when a session
  // ends, since it only changes when the backend restarts.
  const refreshModelStatus = useCallback(async () => {
    try {
      dispatch({ type: "modelStatus", status: await api.getModelStatus() });
    } catch {
      /* the header simply omits the badge when this is unavailable */
    }
  }, [dispatch]);

  useEffect(() => {
    void refreshModelStatus();
  }, [refreshModelStatus]);

  const startLive = useCallback(
    async (coachSpeakerId: string | null) => {
      dispatch({ type: "session/starting" });
      try {
        const { session_id } = await api.startSession({
          session_type: "live",
          coach_speaker_id: coachSpeakerId,
        });
        dispatch({ type: "session/started", sessionId: session_id });
      } catch (error) {
        dispatch({ type: "session/failed", error: (error as Error).message });
      }
    },
    [dispatch],
  );

  const startFile = useCallback(
    async (file: File, coachSpeakerId: string | null) => {
      dispatch({ type: "session/starting" });
      try {
        const { session_id } = await api.startFileSession(file, coachSpeakerId);
        dispatch({ type: "session/started", sessionId: session_id });
      } catch (error) {
        dispatch({ type: "session/failed", error: (error as Error).message });
      }
    },
    [dispatch],
  );

  const stop = useCallback(async () => {
    dispatch({ type: "session/starting" });
    try {
      const response = await api.stopSession();
      dispatch({ type: "session/stopped", report: response.report });
      void refreshModelStatus();
    } catch (error) {
      dispatch({ type: "session/failed", error: (error as Error).message });
    }
  }, [dispatch, refreshModelStatus]);

  return (
    <div className="flex h-full flex-col">
      <Header status={state.modelStatus} />

      <main className="flex min-h-0 flex-1 flex-col gap-3 p-4 lg:flex-row">
        <aside className="flex w-full shrink-0 flex-col gap-3 lg:w-80">
          <ControlPanel
            active={state.active}
            busy={state.busy}
            sessionId={state.sessionId}
            connection={connection}
            onStartLive={startLive}
            onStartFile={startFile}
            onStop={stop}
          />

          {state.error && (
            <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
              <div className="flex items-start justify-between gap-2">
                <span>{state.error}</span>
                <button
                  type="button"
                  onClick={() => dispatch({ type: "error/clear" })}
                  className="text-rose-300 hover:text-rose-100"
                  aria-label="Dismiss error"
                >
                  ×
                </button>
              </div>
            </div>
          )}

          {state.active && <Suggestions suggestions={stats.suggestions} />}
          {!state.active && <ModelStatusPanel status={state.modelStatus} />}
        </aside>

        <div className="flex min-h-0 flex-1 flex-col gap-3">
          {state.active ? (
            <>
              <StatsBanner stats={stats} />
              <div className="grid min-h-0 flex-1 gap-3 xl:grid-cols-2">
                <LiveTranscript turns={state.turns} partials={state.partials} />
                <div className="flex flex-col gap-3">
                  <GrowTimeline turns={state.turns} />
                  <EmotionChart turns={state.turns} />
                </div>
              </div>
            </>
          ) : (
            <div className="scroll-panel min-h-0 flex-1">
              <SessionReport report={state.report} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

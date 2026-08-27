/** REST client for the FastAPI backend. All paths go through the /api proxy. */

import type {
  AudioDevice,
  ModelStatusPayload,
  SessionStatus,
  StopSessionResponse,
} from "../types";

const BASE = "/api";

export class BackendError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new BackendError(
      "Cannot reach the backend. Is it running on http://localhost:8000?",
    );
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      /* body was not JSON; keep the status text */
    }
    throw new BackendError(detail);
  }

  return (await response.json()) as T;
}

export const checkHealth = () =>
  request<{ status: string; session_active: boolean }>("/health");

export const getModelStatus = () =>
  request<ModelStatusPayload>("/model-status");

export const getSessionStatus = () =>
  request<SessionStatus>("/session/status");

export const getAudioDevices = () =>
  request<{ devices: AudioDevice[] }>("/devices/audio");

export const startSession = (body: {
  session_type?: string;
  device_index?: number | null;
  coach_speaker_id?: string | null;
  /** Replay mode only: path to a stored transcript on the server. */
  transcript_path?: string | null;
}) =>
  request<{ session_id: string; status: string }>("/session/start", {
    method: "POST",
    body: JSON.stringify({ session_type: "live", ...body }),
  });

export const stopSession = () =>
  request<StopSessionResponse>("/session/stop", { method: "POST" });

/** File mode. Sends multipart, so Content-Type must be left to the browser. */
export async function startFileSession(
  file: File,
  coachSpeakerId?: string | null,
): Promise<{ session_id: string; status: string; filename: string }> {
  const form = new FormData();
  form.append("file", file);
  const query = coachSpeakerId
    ? `?coach_speaker_id=${encodeURIComponent(coachSpeakerId)}`
    : "";

  const response = await fetch(`${BASE}/session/start/file${query}`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      /* keep status text */
    }
    throw new BackendError(detail);
  }
  return response.json();
}

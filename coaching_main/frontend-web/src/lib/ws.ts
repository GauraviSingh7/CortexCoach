/**
 * Native WebSocket feed.
 *
 * This is the whole point of moving off Streamlit: messages are *pushed*
 * and dispatched straight into the reducer, so only the components bound
 * to the changed slice re-render. No polling loop, no full-page rerun, no
 * queue shuttling between threads.
 */

import { useEffect, useRef, useState } from "react";
import type { FeedbackMessage } from "../types";

export type ConnectionState = "connecting" | "open" | "closed";

const WS_PATH = "/ws/feedback";

/** Backoff schedule for reconnect attempts, in milliseconds. */
const RETRY_DELAYS = [500, 1000, 2000, 5000, 10000];

function socketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${WS_PATH}`;
}

/**
 * Subscribe to the feedback stream.
 *
 * `onMessage` is held in a ref so callers can pass an inline closure
 * without forcing the socket to tear down and reconnect on every render.
 */
export function useFeedbackSocket(
  enabled: boolean,
  onMessage: (message: FeedbackMessage) => void,
): ConnectionState {
  const [state, setState] = useState<ConnectionState>("closed");
  const handlerRef = useRef(onMessage);
  handlerRef.current = onMessage;

  useEffect(() => {
    if (!enabled) {
      setState("closed");
      return;
    }

    let socket: WebSocket | null = null;
    let retryTimer: number | undefined;
    let attempt = 0;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      setState("connecting");
      socket = new WebSocket(socketUrl());

      socket.onopen = () => {
        attempt = 0;
        setState("open");
      };

      socket.onmessage = (event) => {
        try {
          handlerRef.current(JSON.parse(event.data) as FeedbackMessage);
        } catch (error) {
          console.error("Malformed feedback message", error);
        }
      };

      socket.onerror = () => socket?.close();

      socket.onclose = () => {
        setState("closed");
        if (disposed) return;
        const delay = RETRY_DELAYS[Math.min(attempt, RETRY_DELAYS.length - 1)];
        attempt += 1;
        retryTimer = window.setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      disposed = true;
      window.clearTimeout(retryTimer);
      // Drop the close handler first so teardown does not schedule a retry.
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
  }, [enabled]);

  return state;
}

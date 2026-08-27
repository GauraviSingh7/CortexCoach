"""Backend transport for the dashboard: REST calls and the WebSocket feed."""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, Dict, List, Optional

import requests
import websocket

API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/feedback"

#: Stopping a session can be slow the first time (ChromaDB downloads an
#: embedding model), so this is deliberately generous.
STOP_TIMEOUT_SECONDS = 180


class WebSocketClient:
    """Background WebSocket reader with a thread-safe message queue."""

    def __init__(self, url: str = WS_URL):
        self.url = url
        self.ws: Optional[websocket.WebSocketApp] = None
        self.connected = False
        self.message_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

    def connect(self) -> bool:
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
            )
            threading.Thread(target=self.ws.run_forever, daemon=True).start()
            self.connected = True
            return True
        except Exception as exc:
            print(f"WebSocket connection failed: {exc}")
            return False

    def _on_open(self, ws) -> None:
        self.connected = True

    def _on_message(self, ws, message: str) -> None:
        try:
            self.message_queue.put(json.loads(message))
        except Exception as exc:
            print(f"Error decoding websocket message: {exc}")

    def _on_error(self, ws, error) -> None:
        print(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self.connected = False

    def drain(self) -> List[Dict[str, Any]]:
        """Pop every pending message."""
        messages: List[Dict[str, Any]] = []
        while True:
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages

    # Backwards-compatible alias.
    get_messages = drain


class BackendError(Exception):
    """Raised when the backend rejects a request."""


def _detail(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text


def start_session(session_type: str = "live", **kwargs) -> Dict[str, Any]:
    """Start a session. Extra kwargs (e.g. transcript_path) pass through."""
    response = requests.post(
        f"{API_BASE_URL}/session/start",
        json={"session_type": session_type, **kwargs},
        timeout=10,
    )
    if response.status_code != 200:
        raise BackendError(_detail(response))
    return response.json()


def stop_session() -> Dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/session/stop", timeout=STOP_TIMEOUT_SECONDS
    )
    if response.status_code != 200:
        raise BackendError(_detail(response))
    return response.json()


def get_session_status() -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_BASE_URL}/session/status", timeout=2)
        return response.json() if response.status_code == 200 else None
    except requests.RequestException:
        return None


def check_health() -> bool:
    """True when the backend answers its health endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_model_status() -> Optional[Dict[str, Any]]:
    """Per-model state and blocking reasons, for the diagnostics panel."""
    try:
        response = requests.get(f"{API_BASE_URL}/model-status", timeout=5)
        return response.json() if response.status_code == 200 else None
    except requests.RequestException:
        return None

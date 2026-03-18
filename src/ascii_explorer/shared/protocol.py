"""Wire protocol for ascii-explorer.

All messages are newline-delimited JSON sent over raw TCP streams.

Client -> Server message types:
    MSG_JOIN   {"type": "join",  "name": "<player_name>"}
    MSG_MOVE   {"type": "move",  "dir": "up|down|left|right"}
    MSG_CHAT   {"type": "chat",  "msg": "<text>"}

Server -> Client message types:
    MSG_INIT   {"type": "init",  "player_id": int, "color_id": int,
                "x": int, "y": int, "map": List[str]}
    MSG_STATE  {"type": "state", "players": List[PlayerDict],
                "items": List[List[int]]}
    MSG_EVENT  {"type": "event", "msg": "<text>"}
    MSG_ERROR  {"type": "error", "msg": "<text>"}

PlayerDict schema:
    {"id": int, "name": str, "x": int, "y": int,
     "score": int, "color_id": int}
"""

from __future__ import annotations

import json
import socket
from typing import Any, Iterator

MSG_JOIN: str = "join"
MSG_MOVE: str = "move"
MSG_CHAT: str = "chat"
MSG_INIT: str = "init"
MSG_STATE: str = "state"
MSG_EVENT: str = "event"
MSG_ERROR: str = "error"

# Valid movement directions
DIRECTIONS: list[str] = ["up", "down", "left", "right"]

# Direction -> (dx, dy) delta
DIR_DELTA: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def encode(msg: dict[str, Any]) -> bytes:
    """Serialize a message dict to a newline-terminated UTF-8 byte string."""
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: str) -> dict[str, Any]:
    """Deserialize a single JSON line to a message dict."""
    return json.loads(line.strip())  # type: ignore[no-any-return]


def make_init(
    player_id: int,
    color_id: int,
    x: int,
    y: int,
    map_rows: list[str],
) -> dict[str, Any]:
    """Build an MSG_INIT payload."""
    return {
        "type": MSG_INIT,
        "player_id": player_id,
        "color_id": color_id,
        "x": x,
        "y": y,
        "map": map_rows,
    }


def make_state(
    players: list[dict[str, Any]],
    items: list[list[int]],
) -> dict[str, Any]:
    """Build an MSG_STATE payload."""
    return {
        "type": MSG_STATE,
        "players": players,
        "items": items,
    }


def make_event(msg: str) -> dict[str, Any]:
    """Build an MSG_EVENT payload."""
    return {"type": MSG_EVENT, "msg": msg}


def make_error(msg: str) -> dict[str, Any]:
    """Build an MSG_ERROR payload."""
    return {"type": MSG_ERROR, "msg": msg}


class SocketReader:
    """Wraps a TCP socket and yields complete newline-delimited JSON messages.

    TCP is a stream protocol; message boundaries are not guaranteed. This class
    buffers incoming bytes and yields one decoded dict per complete line.

    Usage::

        reader = SocketReader(conn)
        for msg in reader:
            handle(msg)
    """

    _RECV_SIZE: int = 4096

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf: str = ""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self._read_messages()

    def _read_messages(self) -> Iterator[dict[str, Any]]:
        """Yield decoded message dicts until the socket closes."""
        while True:
            try:
                chunk = self._sock.recv(self._RECV_SIZE)
            except OSError:
                break
            if not chunk:
                break
            self._buf += chunk.decode("utf-8", errors="replace")
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    yield decode(line)
                except (json.JSONDecodeError, ValueError):
                    # Malformed message -- skip silently
                    continue

    def send(self, msg: dict[str, Any]) -> None:
        """Encode and send a message dict over the socket."""
        try:
            self._sock.sendall(encode(msg))
        except OSError:
            pass

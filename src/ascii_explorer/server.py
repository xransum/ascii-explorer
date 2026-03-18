"""TCP game server for ascii-explorer.

Architecture:
    - Single authoritative GameState protected by a threading.Lock.
    - One thread per connected client (up to MAX_CLIENTS).
    - After every valid action the full state is broadcast to all clients.

Running::

    uv run ascii-server
    uv run ascii-server --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

from ascii_explorer.shared.map import (
    TILE_FLOOR,
    TILE_GEM,
    Point,
    generate_map,
    get_tile,
    grid_to_strings,
    set_tile,
)
from ascii_explorer.shared.protocol import (
    DIR_DELTA,
    DIRECTIONS,
    MSG_CHAT,
    MSG_JOIN,
    MSG_MOVE,
    SocketReader,
    make_error,
    make_event,
    make_init,
    make_state,
)

DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 9000
MAX_CLIENTS: int = 10
MAP_WIDTH: int = 120
MAP_HEIGHT: int = 120

# Color IDs 1-10 (matched to curses color pairs on the client)
_COLOR_IDS: list[int] = list(range(1, 11))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ascii-explorer.server")


@dataclass
class Player:
    """Represents a connected player in the game."""

    pid: int
    name: str
    x: int
    y: int
    score: int = 0
    color_id: int = 1
    conn: socket.socket | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for wire transmission."""
        return {
            "id": self.pid,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "score": self.score,
            "color_id": self.color_id,
        }


class GameState:
    """Authoritative game state. All mutations must hold the lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        log.info("Generating map %dx%d ...", MAP_WIDTH, MAP_HEIGHT)
        self._grid, self._spawns, self._items = generate_map(
            width=MAP_WIDTH,
            height=MAP_HEIGHT,
        )
        log.info(
            "Map generated. Rooms connected. %d gems placed. %d spawn points.",
            len(self._items),
            len(self._spawns),
        )
        self._map_rows: list[str] = grid_to_strings(self._grid)
        self._players: dict[int, Player] = {}
        self._next_pid: int = 1
        self._used_colors: list[int] = []
        self._spawn_index: int = 0
        # All connected sockets for broadcasting
        self._connections: dict[int, socket.socket] = {}

    # ------------------------------------------------------------------
    # Internal helpers (call only while holding lock)
    # ------------------------------------------------------------------

    def _next_color(self) -> int:
        available = [c for c in _COLOR_IDS if c not in self._used_colors]
        if available:
            color = available[0]
        else:
            color = (self._next_pid % len(_COLOR_IDS)) + 1
        self._used_colors.append(color)
        return color

    def _next_spawn(self) -> Point:
        if self._spawns:
            pt = self._spawns[self._spawn_index % len(self._spawns)]
            self._spawn_index += 1
            return pt
        # Fallback: center of map
        return (MAP_WIDTH // 2, MAP_HEIGHT // 2)

    def _occupied_positions(self) -> set[Point]:
        return {(p.x, p.y) for p in self._players.values()}

    def _broadcast_unlocked(self, msg: dict[str, Any]) -> None:
        """Send a message to all connected clients. Must be called under lock."""
        from ascii_explorer.shared.protocol import encode

        data = encode(msg)
        dead: list[int] = []
        for pid, conn in self._connections.items():
            try:
                conn.sendall(data)
            except OSError:
                dead.append(pid)
        for pid in dead:
            self._connections.pop(pid, None)

    def _state_payload(self) -> dict[str, Any]:
        """Build the current state dict. Must be called under lock."""
        players_list = [p.to_dict() for p in self._players.values()]
        items_list = [[x, y] for x, y in self._items]
        return make_state(players_list, items_list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def map_rows(self) -> list[str]:
        """The map as a list of strings (immutable after generation)."""
        return self._map_rows

    def add_player(self, name: str, conn: socket.socket) -> tuple[Player, list[str]]:
        """Register a new player. Returns (player, map_rows).

        Also broadcasts a join event to all existing players.
        """
        with self._lock:
            pid = self._next_pid
            self._next_pid += 1
            color = self._next_color()
            x, y = self._next_spawn()

            player = Player(
                pid=pid,
                name=name,
                x=x,
                y=y,
                color_id=color,
                conn=conn,
            )
            self._players[pid] = player
            self._connections[pid] = conn

            # Broadcast join event to everyone (including the new player
            # whose connection is now registered)
            self._broadcast_unlocked(make_event(f"{name} joined the game."))
            self._broadcast_unlocked(self._state_payload())

            log.info("Player %d (%s) joined at (%d, %d).", pid, name, x, y)
            return player, self._map_rows

    def remove_player(self, pid: int) -> None:
        """Unregister a player and broadcast a leave event."""
        with self._lock:
            player = self._players.pop(pid, None)
            self._connections.pop(pid, None)
            if player is not None:
                if player.color_id in self._used_colors:
                    self._used_colors.remove(player.color_id)
                self._broadcast_unlocked(make_event(f"{player.name} left the game."))
                self._broadcast_unlocked(self._state_payload())
                log.info("Player %d (%s) disconnected.", pid, player.name)

    def move_player(self, pid: int, direction: str) -> tuple[bool, list[str]]:
        """Attempt to move a player in the given direction.

        Returns (success, [event_messages]).
        On success, broadcasts updated state to all clients.
        """
        events: list[str] = []

        with self._lock:
            player = self._players.get(pid)
            if player is None:
                return False, []

            if direction not in DIRECTIONS:
                return False, []

            dx, dy = DIR_DELTA[direction]
            nx, ny = player.x + dx, player.y + dy

            # Collision: wall
            if not self._is_walkable_unlocked(nx, ny):
                return False, []

            # Collision: another player
            if (nx, ny) in self._occupied_positions() - {(player.x, player.y)}:
                return False, []

            # Move player
            player.x = nx
            player.y = ny

            # Gem pickup
            tile = get_tile(self._grid, nx, ny)
            if tile == TILE_GEM:
                player.score += 1
                set_tile(self._grid, nx, ny, TILE_FLOOR)
                self._items.discard((nx, ny))
                # Rebuild map rows for this row only
                self._map_rows[ny] = "".join(self._grid[ny])
                events.append(f"{player.name} picked up a gem! (score: {player.score})")

            self._broadcast_unlocked(self._state_payload())
            for evt in events:
                self._broadcast_unlocked(make_event(evt))

        return True, events

    def broadcast_chat(self, pid: int, msg: str) -> None:
        """Broadcast a chat message from the given player to all clients."""
        with self._lock:
            player = self._players.get(pid)
            if player is None:
                return
            # Truncate to prevent abuse
            safe_msg = msg[:200].strip()
            if safe_msg:
                self._broadcast_unlocked(make_event(f"[{player.name}]: {safe_msg}"))

    def _is_walkable_unlocked(self, x: int, y: int) -> bool:
        """Check walkability without acquiring the lock (caller must hold it)."""
        from ascii_explorer.shared.map import is_walkable

        return is_walkable(self._grid, x, y)

    def player_count(self) -> int:
        """Return the current number of connected players."""
        with self._lock:
            return len(self._players)


def handle_client(
    conn: socket.socket,
    addr: tuple[str, int],
    state: GameState,
) -> None:
    """Handle the full lifecycle of a single client connection."""
    log.info("New connection from %s:%d", addr[0], addr[1])
    reader = SocketReader(conn)
    player: Player | None = None

    try:
        # --- Handshake: wait for MSG_JOIN ---
        for msg in reader:
            if msg.get("type") != MSG_JOIN:
                reader.send(make_error("First message must be 'join'."))
                continue

            raw_name = msg.get("name", "").strip()
            if not raw_name:
                reader.send(make_error("Name cannot be empty."))
                continue

            name = raw_name[:20]  # cap name length
            player, map_rows = state.add_player(name, conn)

            # Send init to the new player
            reader.send(
                make_init(
                    player_id=player.pid,
                    color_id=player.color_id,
                    x=player.x,
                    y=player.y,
                    map_rows=map_rows,
                )
            )
            break

        if player is None:
            log.warning("Client %s:%d never sent a valid join.", addr[0], addr[1])
            return

        # --- Main message loop ---
        for msg in reader:
            msg_type = msg.get("type")

            if msg_type == MSG_MOVE:
                direction = msg.get("dir", "")
                ok, _ = state.move_player(player.pid, direction)
                if not ok:
                    reader.send(make_error("Invalid move."))

            elif msg_type == MSG_CHAT:
                text = msg.get("msg", "")
                state.broadcast_chat(player.pid, text)

            else:
                reader.send(make_error(f"Unknown message type: {msg_type!r}"))

    except Exception as exc:
        log.exception("Unhandled error for client %s:%d: %s", addr[0], addr[1], exc)
    finally:
        conn.close()
        if player is not None:
            state.remove_player(player.pid)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ascii-explorer game server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ascii-explorer server."""
    args = _parse_args(argv)
    state = GameState()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(MAX_CLIENTS)

    log.info(
        "Server listening on %s:%d (max %d clients)",
        args.host,
        args.port,
        MAX_CLIENTS,
    )

    try:
        while True:
            conn, addr = server_sock.accept()

            if state.player_count() >= MAX_CLIENTS:
                log.warning(
                    "Rejecting connection from %s:%d -- server full.", addr[0], addr[1]
                )
                try:
                    from ascii_explorer.shared.protocol import encode

                    conn.sendall(encode(make_error("Server is full (max 10 players).")))
                except OSError:
                    pass
                conn.close()
                continue

            thread = threading.Thread(
                target=handle_client,
                args=(conn, addr, state),
                daemon=True,
            )
            thread.start()

    except KeyboardInterrupt:
        log.info("Server shutting down.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()

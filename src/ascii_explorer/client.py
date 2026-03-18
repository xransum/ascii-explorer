"""Terminal curses client for ascii-explorer.

Architecture:
    - Main thread: curses render loop at ~20fps.
    - Input thread: reads keypresses, handles chat mode, sends to server.
    - Receive thread: reads server messages, updates shared state buffer.

Running::

    uv run ascii-client --host 127.0.0.1 --port 9000 --name Alice
"""

from __future__ import annotations

import argparse
import curses
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from ascii_explorer.shared.map import (
    TILE_DOOR,
    TILE_FLOOR,
    TILE_GEM,
    TILE_WALL,
    strings_to_grid,
)
from ascii_explorer.shared.protocol import (
    MSG_ERROR,
    MSG_EVENT,
    MSG_INIT,
    MSG_STATE,
    SocketReader,
    encode,
    make_event,
)

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 9000
TARGET_FPS: int = 20
EVENT_LOG_SIZE: int = 50  # max lines kept in the event log buffer
STATUS_HEIGHT: int = 1  # rows for the status bar
EVENT_HEIGHT: int = 3  # rows for the event log panel
MIN_VIEWPORT_W: int = 40
MIN_VIEWPORT_H: int = 10

#
# Pair 0  = default (curses built-in, cannot be changed)
# Pairs 1-10 = player colors
# Pair 11 = gem (bright yellow)
# Pair 12 = wall (dark / dim)
# Pair 13 = door (yellow)
# Pair 14 = event log text
# Pair 15 = status bar

_PAIR_GEM: int = 11
_PAIR_WALL: int = 12
_PAIR_DOOR: int = 13
_PAIR_EVENT: int = 14
_PAIR_STATUS: int = 15
_PAIR_CHAT_PROMPT: int = 16

# Player color palette (curses color constants assigned at init time)
_PLAYER_COLORS = [
    curses.COLOR_RED,
    curses.COLOR_GREEN,
    curses.COLOR_YELLOW,
    curses.COLOR_BLUE,
    curses.COLOR_MAGENTA,
    curses.COLOR_CYAN,
    curses.COLOR_WHITE,
    # Bright variants via bold attribute -- same base colors
    curses.COLOR_RED,
    curses.COLOR_GREEN,
    curses.COLOR_CYAN,
]

# color_id 1-10 maps to index 0-9 in _PLAYER_COLORS
# Pairs 1-10 are initialized in _init_colors()


@dataclass
class PlayerInfo:
    """Snapshot of a remote player."""

    pid: int
    name: str
    x: int
    y: int
    score: int
    color_id: int


@dataclass
class ClientState:
    """All mutable state shared between render, input, and receive threads."""

    lock: threading.Lock = field(default_factory=threading.Lock)

    # Set after MSG_INIT
    player_id: int = -1
    color_id: int = 1
    self_x: int = 0
    self_y: int = 0
    grid: list[list[str]] = field(default_factory=lambda: [])

    # Updated by MSG_STATE
    players: dict[int, PlayerInfo] = field(default_factory=lambda: {})
    items: set[tuple[int, int]] = field(default_factory=lambda: set())

    # Event log (thread-safe append via lock)
    events: deque[str] = field(default_factory=lambda: deque(maxlen=EVENT_LOG_SIZE))

    # Chat input state (owned by input thread, read by render thread)
    in_chat_mode: bool = False
    chat_buf: str = ""

    # Signal that the receive thread has died (connection lost)
    disconnected: bool = False
    disconnect_reason: str = ""

    # True once MSG_INIT received
    initialized: bool = False


def _init_colors() -> None:
    """Initialize curses color pairs."""
    curses.start_color()
    curses.use_default_colors()

    # Player colors: pairs 1-10
    for i, base_color in enumerate(_PLAYER_COLORS):
        curses.init_pair(i + 1, base_color, -1)

    # Special tile colors
    curses.init_pair(_PAIR_GEM, curses.COLOR_YELLOW, -1)
    curses.init_pair(_PAIR_WALL, curses.COLOR_WHITE, -1)
    curses.init_pair(_PAIR_DOOR, curses.COLOR_YELLOW, -1)
    curses.init_pair(_PAIR_EVENT, curses.COLOR_CYAN, -1)
    curses.init_pair(_PAIR_STATUS, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(_PAIR_CHAT_PROMPT, curses.COLOR_YELLOW, -1)


def _player_attr(color_id: int, is_self: bool = False) -> int:
    """Return curses attribute for a player glyph."""
    pair = curses.color_pair(max(1, min(color_id, 10)))
    if color_id >= 8 or is_self:
        return pair | curses.A_BOLD
    return pair


def _safe_addch(
    win: curses.window,
    y: int,
    x: int,
    ch: str,
    attr: int = 0,
) -> None:
    """Add a character, ignoring out-of-bounds or end-of-line errors."""
    try:
        win.addch(y, x, ord(ch) if len(ch) == 1 else ord("?"), attr)
    except curses.error:
        pass


def _safe_addstr(
    win: curses.window,
    y: int,
    x: int,
    text: str,
    attr: int = 0,
) -> None:
    """Add a string, ignoring errors from writing to the last cell."""
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def _draw_map(
    win: curses.window,
    state: ClientState,
    vp_h: int,
    vp_w: int,
) -> None:
    """Draw the visible viewport of the map onto win.

    The viewport is centered on the local player.
    """
    grid = state.grid
    if not grid:
        return

    map_h = len(grid)
    map_w = len(grid[0]) if map_h > 0 else 0

    # Camera center: clamp so viewport stays inside map bounds
    cam_x = max(vp_w // 2, min(state.self_x, map_w - vp_w // 2 - 1))
    cam_y = max(vp_h // 2, min(state.self_y, map_h - vp_h // 2 - 1))

    start_map_x = cam_x - vp_w // 2
    start_map_y = cam_y - vp_h // 2

    # Build a quick lookup of player positions
    player_at: dict[tuple[int, int], PlayerInfo] = {
        (p.x, p.y): p for p in state.players.values()
    }

    for vy in range(vp_h):
        mx = start_map_x
        my = start_map_y + vy
        for vx in range(vp_w):
            if mx < 0 or my < 0 or mx >= map_w or my >= map_h:
                _safe_addch(win, vy, vx, " ")
                mx += 1
                continue

            pos: tuple[int, int] = (mx, my)

            # Player glyph takes priority
            if pos in player_at:
                p = player_at[pos]
                is_self = p.pid == state.player_id
                attr = _player_attr(p.color_id, is_self)
                _safe_addch(win, vy, vx, "@", attr)
            else:
                tile = grid[my][mx]
                if tile == TILE_WALL:
                    attr = curses.color_pair(_PAIR_WALL) | curses.A_DIM
                    _safe_addch(win, vy, vx, "#", attr)
                elif tile == TILE_GEM:
                    attr = curses.color_pair(_PAIR_GEM) | curses.A_BOLD
                    _safe_addch(win, vy, vx, "*", attr)
                elif tile == TILE_DOOR:
                    attr = curses.color_pair(_PAIR_DOOR)
                    _safe_addch(win, vy, vx, "+", attr)
                elif tile == TILE_FLOOR:
                    _safe_addch(win, vy, vx, ".", curses.A_DIM)
                else:
                    _safe_addch(win, vy, vx, tile)

            mx += 1


def _draw_event_log(
    win: curses.window,
    events: deque[str],
    panel_h: int,
    panel_w: int,
) -> None:
    """Draw the most recent event log lines onto win."""
    win.erase()
    attr = curses.color_pair(_PAIR_EVENT)
    lines = list(events)[-(panel_h):]
    for i, line in enumerate(lines):
        _safe_addstr(win, i, 0, line[: panel_w - 1], attr)


def _draw_status(
    win: curses.window,
    state: ClientState,
    panel_w: int,
) -> None:
    """Draw the status bar (or chat prompt) onto win."""
    win.erase()
    attr = curses.color_pair(_PAIR_STATUS)

    if state.in_chat_mode:
        prompt = f"Say: {state.chat_buf}_"
        _safe_addstr(
            win, 0, 0, prompt[: panel_w - 1], curses.color_pair(_PAIR_CHAT_PROMPT)
        )
    else:
        me = state.players.get(state.player_id)
        name = me.name if me else "?"
        score = me.score if me else 0
        x, y = state.self_x, state.self_y
        online = len(state.players)
        status = (
            f" {name}  Score: {score}  Pos: ({x},{y})"
            f"  Players online: {online}"
            f"  [WASD/arrows=move  /=chat  q=quit]"
        )
        _safe_addstr(win, 0, 0, " " * panel_w, attr)
        _safe_addstr(win, 0, 0, status[: panel_w - 1], attr)


def receive_thread(sock: socket.socket, state: ClientState) -> None:
    """Read server messages and update ClientState."""
    reader = SocketReader(sock)
    try:
        for msg in reader:
            msg_type = msg.get("type")

            if msg_type == MSG_INIT:
                with state.lock:
                    state.player_id = msg["player_id"]
                    state.color_id = msg["color_id"]
                    state.self_x = msg["x"]
                    state.self_y = msg["y"]
                    state.grid = strings_to_grid(msg["map"])
                    state.initialized = True

            elif msg_type == MSG_STATE:
                with state.lock:
                    new_players: dict[int, PlayerInfo] = {}
                    for pd in msg.get("players", []):
                        info = PlayerInfo(
                            pid=pd["id"],
                            name=pd["name"],
                            x=pd["x"],
                            y=pd["y"],
                            score=pd["score"],
                            color_id=pd["color_id"],
                        )
                        new_players[info.pid] = info
                        if info.pid == state.player_id:
                            state.self_x = info.x
                            state.self_y = info.y
                    state.players = new_players
                    state.items = {(item[0], item[1]) for item in msg.get("items", [])}

            elif msg_type == MSG_EVENT:
                with state.lock:
                    state.events.append(msg.get("msg", ""))

            elif msg_type == MSG_ERROR:
                with state.lock:
                    state.events.append(f"[!] {msg.get('msg', '')}")

    except Exception:
        pass
    finally:
        with state.lock:
            state.disconnected = True
            state.disconnect_reason = "Connection to server lost."


_KEY_MOVE: dict[int, str] = {
    ord("w"): "up",
    ord("W"): "up",
    curses.KEY_UP: "up",
    ord("s"): "down",
    ord("S"): "down",
    curses.KEY_DOWN: "down",
    ord("a"): "left",
    ord("A"): "left",
    curses.KEY_LEFT: "left",
    ord("d"): "right",
    ord("D"): "right",
    curses.KEY_RIGHT: "right",
}


def input_thread(
    stdscr: curses.window,
    sock: socket.socket,
    state: ClientState,
) -> None:
    """Read keypresses and send messages to the server."""
    stdscr.nodelay(True)
    stdscr.keypad(True)

    while True:
        with state.lock:
            disconnected = state.disconnected
            in_chat = state.in_chat_mode

        if disconnected:
            break

        try:
            key = stdscr.getch()
        except curses.error:
            time.sleep(0.01)
            continue

        if key == curses.ERR:
            time.sleep(0.01)
            continue

        if in_chat:
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                # Send chat message
                with state.lock:
                    msg_text = state.chat_buf.strip()
                    state.chat_buf = ""
                    state.in_chat_mode = False
                if msg_text:
                    try:
                        sock.sendall(encode({"type": "chat", "msg": msg_text}))
                    except OSError:
                        break
            elif key == 27:  # ESC
                with state.lock:
                    state.chat_buf = ""
                    state.in_chat_mode = False
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                with state.lock:
                    state.chat_buf = state.chat_buf[:-1]
            elif 32 <= key < 127:
                with state.lock:
                    if len(state.chat_buf) < 200:
                        state.chat_buf += chr(key)
        else:
            if key in _KEY_MOVE:
                direction = _KEY_MOVE[key]
                try:
                    sock.sendall(encode({"type": "move", "dir": direction}))
                except OSError:
                    break
            elif key == ord("/"):
                with state.lock:
                    state.in_chat_mode = True
                    state.chat_buf = ""
            elif key in (ord("q"), ord("Q")):
                with state.lock:
                    state.disconnected = True
                    state.disconnect_reason = "You quit."
                break


def run(stdscr: curses.window, sock: socket.socket, state: ClientState) -> None:
    """Main curses loop: renders the game at TARGET_FPS until disconnected."""
    _init_colors()
    curses.curs_set(0)

    # Start receive thread
    recv_t = threading.Thread(target=receive_thread, args=(sock, state), daemon=True)
    recv_t.start()

    # Start input thread
    input_t = threading.Thread(
        target=input_thread, args=(stdscr, sock, state), daemon=True
    )
    input_t.start()

    frame_duration = 1.0 / TARGET_FPS

    while True:
        with state.lock:
            disconnected = state.disconnected
            reason = state.disconnect_reason
            initialized = state.initialized

        if disconnected:
            # Show disconnect message before exiting
            stdscr.erase()
            _safe_addstr(stdscr, 0, 0, reason or "Disconnected.", curses.A_BOLD)
            stdscr.refresh()
            time.sleep(2.0)
            break

        if not initialized:
            stdscr.erase()
            _safe_addstr(stdscr, 0, 0, "Connecting to server...")
            stdscr.refresh()
            time.sleep(frame_duration)
            continue

        term_h, term_w = stdscr.getmaxyx()

        # Layout: viewport fills top, event log in the middle, status at bottom
        vp_h = max(MIN_VIEWPORT_H, term_h - EVENT_HEIGHT - STATUS_HEIGHT)
        vp_w = max(MIN_VIEWPORT_W, term_w)

        # Create sub-windows (erase and recreate each frame to handle resize)
        try:
            map_win = curses.newwin(vp_h, vp_w, 0, 0)
            evt_win = curses.newwin(EVENT_HEIGHT, term_w, vp_h, 0)
            status_win = curses.newwin(STATUS_HEIGHT, term_w, vp_h + EVENT_HEIGHT, 0)
        except curses.error:
            time.sleep(frame_duration)
            continue

        with state.lock:
            _draw_map(map_win, state, vp_h, vp_w)
            _draw_event_log(evt_win, state.events, EVENT_HEIGHT, term_w)
            _draw_status(status_win, state, term_w)

        map_win.noutrefresh()
        evt_win.noutrefresh()
        status_win.noutrefresh()
        curses.doupdate()

        time.sleep(frame_duration)


def connect_and_join(
    host: str, port: int, name: str
) -> tuple[socket.socket, ClientState]:
    """Connect to the server, send join, and return (socket, initial state)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    # Send join message
    sock.sendall(encode({"type": "join", "name": name}))

    state = ClientState()
    return sock, state


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ascii-explorer game client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--name", required=True, help="Your player name (max 20 chars)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ascii-explorer client."""
    args = _parse_args(argv)

    try:
        sock, state = connect_and_join(args.host, args.port, args.name)
    except OSError as exc:
        print(f"Could not connect to {args.host}:{args.port} -- {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        curses.wrapper(run, sock, state)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.sendall(encode(make_event("quit")))
        except OSError:
            pass
        sock.close()


if __name__ == "__main__":
    main()

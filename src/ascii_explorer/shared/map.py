"""Procedural map generation for ascii-explorer.

Two-pass hybrid approach:
  1. BSP (Binary Space Partitioning) -- carves structured rooms connected by corridors.
  2. Cellular Automata smoothing -- roughens room/corridor edges for an organic feel.

Post-processing validates full connectivity via BFS flood fill, patches orphaned
regions, places gem items, and selects spread-apart spawn points.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

TILE_WALL: str = "#"
TILE_FLOOR: str = "."
TILE_DOOR: str = "+"
TILE_GEM: str = "*"

Grid = List[List[str]]
Point = Tuple[int, int]

_MIN_LEAF_SIZE: int = 10  # minimum BSP partition size (width or height)
_MIN_ROOM_SIZE: int = 5  # minimum room interior dimension


class BSPNode:
    """A node in the Binary Space Partition tree."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left: BSPNode | None = None
        self.right: BSPNode | None = None
        # Room carved inside this leaf (x1, y1, x2, y2) inclusive
        self.room: tuple[int, int, int, int] | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def split(self, rng: random.Random) -> bool:
        """Attempt to split this node. Returns True if split succeeded."""
        if not self.is_leaf:
            return False

        # Decide split direction: prefer the longer axis, add some randomness
        split_h = rng.random() > 0.5
        if self.w > self.h and self.w / self.h >= 1.25:
            split_h = False
        elif self.h > self.w and self.h / self.w >= 1.25:
            split_h = True

        max_size = (self.h if split_h else self.w) - _MIN_LEAF_SIZE
        if max_size <= _MIN_LEAF_SIZE:
            return False  # too small to split

        split_pos = rng.randint(_MIN_LEAF_SIZE, max_size)

        if split_h:
            self.left = BSPNode(self.x, self.y, self.w, split_pos)
            self.right = BSPNode(self.x, self.y + split_pos, self.w, self.h - split_pos)
        else:
            self.left = BSPNode(self.x, self.y, split_pos, self.h)
            self.right = BSPNode(self.x + split_pos, self.y, self.w - split_pos, self.h)

        return True

    def get_leaves(self) -> list[BSPNode]:
        """Return all leaf nodes under this node."""
        if self.is_leaf:
            return [self]
        leaves: list[BSPNode] = []
        if self.left is not None:
            leaves.extend(self.left.get_leaves())
        if self.right is not None:
            leaves.extend(self.right.get_leaves())
        return leaves

    def get_room_center(self) -> Point | None:
        """Return the center of this node's room, or None if no room."""
        if self.room is None:
            return None
        x1, y1, x2, y2 = self.room
        return ((x1 + x2) // 2, (y1 + y2) // 2)


def _build_bsp_tree(
    node: BSPNode, rng: random.Random, depth: int = 0, max_depth: int = 6
) -> None:
    """Recursively split the BSP tree up to max_depth."""
    if depth >= max_depth:
        return
    if node.split(rng):
        assert node.left is not None
        assert node.right is not None
        _build_bsp_tree(node.left, rng, depth + 1, max_depth)
        _build_bsp_tree(node.right, rng, depth + 1, max_depth)


def _carve_rooms(root: BSPNode, grid: Grid, rng: random.Random) -> None:
    """Carve a room inside each BSP leaf."""
    for leaf in root.get_leaves():
        # Room fits inside the leaf with at least 1-tile wall padding
        room_w = rng.randint(_MIN_ROOM_SIZE, max(_MIN_ROOM_SIZE, leaf.w - 2))
        room_h = rng.randint(_MIN_ROOM_SIZE, max(_MIN_ROOM_SIZE, leaf.h - 2))
        room_x = leaf.x + rng.randint(1, max(1, leaf.w - room_w - 1))
        room_y = leaf.y + rng.randint(1, max(1, leaf.h - room_h - 1))

        x2 = min(room_x + room_w - 1, len(grid[0]) - 2)
        y2 = min(room_y + room_h - 1, len(grid) - 2)

        leaf.room = (room_x, room_y, x2, y2)

        for row in range(room_y, y2 + 1):
            for col in range(room_x, x2 + 1):
                grid[row][col] = TILE_FLOOR


def _connect_nodes(node: BSPNode, grid: Grid, rng: random.Random) -> Point | None:
    """Recursively connect sibling rooms with L-shaped corridors.

    Returns a floor point within this subtree for the parent to connect to.
    """
    if node.is_leaf:
        return node.get_room_center()

    left_pt: Point | None = None
    right_pt: Point | None = None

    if node.left is not None:
        left_pt = _connect_nodes(node.left, grid, rng)
    if node.right is not None:
        right_pt = _connect_nodes(node.right, grid, rng)

    if left_pt is not None and right_pt is not None:
        _carve_corridor(grid, left_pt, right_pt)
        return left_pt if rng.random() > 0.5 else right_pt

    return left_pt or right_pt


def _carve_corridor(grid: Grid, a: Point, b: Point) -> None:
    """Carve an L-shaped corridor between two points."""
    ax, ay = a
    bx, by = b
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    # Horizontal segment first, then vertical
    step_x = 1 if bx >= ax else -1
    for cx in range(ax, bx + step_x, step_x):
        if 0 < cx < width - 1 and 0 < ay < height - 1:
            grid[ay][cx] = TILE_FLOOR

    step_y = 1 if by >= ay else -1
    for cy in range(ay, by + step_y, step_y):
        if 0 < bx < width - 1 and 0 < cy < height - 1:
            grid[cy][bx] = TILE_FLOOR


def _cellular_automata_pass(grid: Grid) -> Grid:
    """Single CA smoothing pass.

    Rule: a wall tile becomes floor if it has >= 5 floor neighbours (8-directional).
    This opens up corners and gives organic edges.
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    new_grid: Grid = [row[:] for row in grid]

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if grid[row][col] != TILE_WALL:
                continue
            floor_count = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if grid[nr][nc] == TILE_FLOOR:
                            floor_count += 1
            if floor_count >= 5:
                new_grid[row][col] = TILE_FLOOR

    return new_grid


def _flood_fill(grid: Grid, start: Point) -> set[Point]:
    """BFS flood fill from start, returning all reachable floor tiles."""
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    visited: set[Point] = set()
    queue: deque[Point] = deque([start])
    visited.add(start)

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if (
                0 <= nx < width
                and 0 <= ny < height
                and (nx, ny) not in visited
                and grid[ny][nx] in (TILE_FLOOR, TILE_DOOR, TILE_GEM)
            ):
                visited.add((nx, ny))
                queue.append((nx, ny))

    return visited


def _collect_all_floor_tiles(grid: Grid) -> list[Point]:
    """Return all floor tile positions in the grid."""
    tiles: list[Point] = []
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] == TILE_FLOOR:
                tiles.append((col, row))
    return tiles


def _patch_connectivity(grid: Grid, rng: random.Random) -> None:
    """Ensure all floor tiles are reachable from the first floor tile found.

    Any disconnected floor regions are connected by carving a corridor to the
    main region.
    """
    all_floor = _collect_all_floor_tiles(grid)
    if not all_floor:
        return

    start = all_floor[0]
    reachable = _flood_fill(grid, start)

    # Find unreachable floor tiles
    unreachable = [pt for pt in all_floor if pt not in reachable]

    while unreachable:
        # Pick a random unreachable tile and connect it to a random reachable tile
        target = rng.choice(unreachable)
        anchor = rng.choice(list(reachable))
        _carve_corridor(grid, anchor, target)

        # Re-flood from original start
        reachable = _flood_fill(grid, start)
        unreachable = [
            pt for pt in _collect_all_floor_tiles(grid) if pt not in reachable
        ]


def _place_gems(
    grid: Grid, reachable: set[Point], rng: random.Random, density: float = 0.05
) -> set[Point]:
    """Scatter gem tiles on reachable floor tiles at the given density."""
    candidates = [pt for pt in reachable if grid[pt[1]][pt[0]] == TILE_FLOOR]
    count = max(1, int(len(candidates) * density))
    chosen = rng.sample(candidates, min(count, len(candidates)))
    items: set[Point] = set()
    for x, y in chosen:
        grid[y][x] = TILE_GEM
        items.add((x, y))
    return items


def _pick_spawn_points(
    reachable: set[Point], n: int, rng: random.Random
) -> list[Point]:
    """Select up to n spawn points spread as far apart as possible.

    Uses a greedy max-distance approach: pick a random first point, then
    iteratively pick the point furthest from all already-chosen points.
    """
    candidates = list(reachable)
    if not candidates:
        return []

    rng.shuffle(candidates)
    spawns: list[Point] = [candidates[0]]

    def min_dist_to_spawns(pt: Point) -> float:
        return min((pt[0] - s[0]) ** 2 + (pt[1] - s[1]) ** 2 for s in spawns)

    while len(spawns) < n and len(spawns) < len(candidates):
        best = max(candidates, key=min_dist_to_spawns)
        if best in spawns:
            break
        spawns.append(best)

    return spawns[:n]


def generate_map(
    width: int = 120,
    height: int = 120,
    seed: int | None = None,
    item_density: float = 0.05,
    max_spawns: int = 10,
    ca_passes: int = 3,
) -> tuple[Grid, list[Point], set[Point]]:
    """Generate a 120x120 map using hybrid BSP + Cellular Automata.

    Returns:
        grid:   2D list of tile characters (height rows x width cols).
        spawns: List of up to max_spawns spread-apart floor Points (x, y).
        items:  Set of gem Points (x, y) placed on the map.
    """
    rng = random.Random(seed)

    # Start with all walls
    grid: Grid = [[TILE_WALL] * width for _ in range(height)]

    # --- Pass 1: BSP room carving ---
    root = BSPNode(0, 0, width, height)
    _build_bsp_tree(root, rng, max_depth=6)
    _carve_rooms(root, grid, rng)
    _connect_nodes(root, grid, rng)

    # --- Pass 2: Cellular automata smoothing ---
    for _ in range(ca_passes):
        grid = _cellular_automata_pass(grid)

    # --- Post-processing ---
    _patch_connectivity(grid, rng)

    all_floor = _collect_all_floor_tiles(grid)
    if not all_floor:
        # Fallback: open a central room if generation produced nothing
        cx, cy = width // 2, height // 2
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                grid[cy + dy][cx + dx] = TILE_FLOOR
        all_floor = _collect_all_floor_tiles(grid)

    start_pt = all_floor[0]
    reachable = _flood_fill(grid, start_pt)

    items = _place_gems(grid, reachable, rng, density=item_density)

    # Exclude gem tiles from spawn candidates
    spawn_candidates = reachable - items
    spawns = _pick_spawn_points(spawn_candidates, max_spawns, rng)

    return grid, spawns, items


def grid_to_strings(grid: Grid) -> list[str]:
    """Convert a 2D grid to a list of row strings for wire transmission."""
    return ["".join(row) for row in grid]


def strings_to_grid(rows: list[str]) -> Grid:
    """Convert a list of row strings back to a 2D grid."""
    return [list(row) for row in rows]


def is_walkable(grid: Grid, x: int, y: int) -> bool:
    """Return True if the tile at (x, y) is walkable (floor, door, or gem)."""
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    return grid[y][x] in (TILE_FLOOR, TILE_DOOR, TILE_GEM)


def get_tile(grid: Grid, x: int, y: int) -> str:
    """Return the tile character at (x, y), or TILE_WALL if out of bounds."""
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    if x < 0 or y < 0 or x >= width or y >= height:
        return TILE_WALL
    return grid[y][x]


def set_tile(grid: Grid, x: int, y: int, tile: str) -> None:
    """Set the tile character at (x, y)."""
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    if 0 <= x < width and 0 <= y < height:
        grid[y][x] = tile

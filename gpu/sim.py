"""GPU Conway's Game of Life on a torus, with Taichi.

The simulation is decoupled from any renderer: it owns the grid, steps it on
the GPU, and exposes the current state. A display (main.py) drives it and reads
back the latest frame, so the sim can run faster than the display samples it.

The board wraps on every edge — a glider that leaves one side reappears on the
other, asteroids-style. Both the rules and the paint brush use that wrap.
"""

import json

import numpy as np
import taichi as ti


@ti.data_oriented
class Simulation:
    """A toroidal Game of Life grid. A cell is 1 (alive) or 0 (dead)."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Ping-pong buffers. `a` is always the current generation; step() fills
        # `b` and swaps. Passing the fields as kernel templates keeps the swap
        # explicit and sidesteps stale field capture.
        self.a = ti.field(ti.i32, shape=(width, height))
        self.b = ti.field(ti.i32, shape=(width, height))
        self.image = ti.field(ti.f32, shape=(width, height))
        self.generations = 0

    @ti.kernel
    def _step(self, src: ti.template(), dst: ti.template()):
        for i, j in src:
            n = 0
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
                if di != 0 or dj != 0:
                    # Modulo wraps the neighbor lookup around the torus.
                    n += src[(i + di) % self.width, (j + dj) % self.height]
            alive = src[i, j]
            # B3/S23: born on exactly 3 neighbors, survive on 2 or 3.
            dst[i, j] = 1 if (n == 3) or (n == 2 and alive == 1) else 0

    def step(self):
        self._step(self.a, self.b)
        self.a, self.b = self.b, self.a
        self.generations += 1

    @ti.kernel
    def _paint(self, field: ti.template(), cx: int, cy: int, radius: int, value: int):
        for i, j in field:
            # Toroidal distance, so a brush near an edge wraps to the far side.
            dx = ti.abs(i - cx)
            dx = ti.min(dx, self.width - dx)
            dy = ti.abs(j - cy)
            dy = ti.min(dy, self.height - dy)
            if dx * dx + dy * dy <= radius * radius:
                field[i, j] = value

    def paint(self, cx: int, cy: int, radius: int, value: int):
        """Stamp a filled circle of `value` (1 draws, 0 erases) at a grid cell."""
        self._paint(self.a, cx, cy, radius, value)

    @ti.kernel
    def _clear(self, field: ti.template()):
        for i, j in field:
            field[i, j] = 0

    def clear(self):
        self._clear(self.a)
        self.generations = 0

    def _load(self, arr: np.ndarray):
        self.a.from_numpy(np.ascontiguousarray(arr, dtype=np.int32))
        self.generations = 0

    def seed_random(self, chance: float):
        """Light each cell with probability `chance` in [0, 1]."""
        self._load(np.random.random((self.width, self.height)) < chance)

    def seed_json(self, path: str):
        """Seed from a list of {Item1, Item2} coordinates, centered on the board.

        Mirrors the WinForms `LetThereBeLight((int x, int y)[])` on main: it
        offsets the pattern to the middle of the grid without subtracting the
        pattern's own origin.
        """
        with open(path, encoding="utf-8-sig") as f:
            points = json.load(f)
        xs = [p["Item1"] for p in points]
        ys = [p["Item2"] for p in points]
        ox = self.width // 2 - (max(xs) - min(xs)) // 2
        oy = self.height // 2 - (max(ys) - min(ys)) // 2
        arr = np.zeros((self.width, self.height), dtype=np.int32)
        for x, y in zip(xs, ys):
            arr[(x + ox) % self.width, (y + oy) % self.height] = 1
        self._load(arr)

    def seed_image(self, path: str, threshold: int = 175):
        """Seed from an image: resize to the grid and light bright pixels.

        Mirrors the `age` branch's `LetThereBeLight(Image)` luminance threshold.
        """
        from PIL import Image

        img = Image.open(path).convert("L").resize((self.width, self.height))
        # PIL rows run top-down; the grid's y runs bottom-up, so flip then
        # transpose to land on (x, y) with the image upright.
        lum = np.flipud(np.asarray(img)).T
        self._load(lum >= threshold)

    @ti.kernel
    def _render(self, src: ti.template(), out: ti.template()):
        for i, j in src:
            out[i, j] = ti.cast(src[i, j], ti.f32)

    def render(self):
        """Return the current generation as a float field for display."""
        self._render(self.a, self.image)
        return self.image

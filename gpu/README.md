# Game of Life on the GPU (Taichi)

A GPU implementation of Conway's Game of Life that you can draw into with the
mouse. The board is a torus — cells, brush strokes, and gliders all wrap around
the edges, asteroids-style.

This is the `gpu-taichi` branch. It reimplements the WinForms simulation in
Python with [Taichi](https://www.taichi-lang.org/), which compiles the rules to
the GPU and gives you a window in a few lines. The simulation lives in `sim.py`
and runs independently of the display in `main.py`, so it can advance faster
than the window samples it.

## Requirements

- Python 3.12. Taichi 1.7.4 has no wheel for Python 3.14, so the project pins to
  3.12 in its own virtual environment.
- A GPU. Taichi selects CUDA, Vulkan, or Metal automatically; it falls back to
  the CPU if none is available.

## Setup

```sh
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS / Linux
```

## Run

```sh
.venv/Scripts/python main.py
```

The window opens paused with a centered seed pattern so you can draw before you
start. Press space to run.

## Controls

- `space` — run or pause
- left mouse — draw live cells (hold and drag)
- right mouse — erase (hold and drag)
- `r` — reseed at random
- `j` — reseed from `initialState.json`
- `i` — reseed from `cortana.jpg`
- `c` — clear the board
- `z` / `x` — shrink or grow the brush
- `esc` — quit

## Seeding

The branch keeps the seeds the WinForms app uses and adds mouse drawing:

- Random — light each cell with a fixed probability.
- JSON — load `initialState.json`, a list of `{Item1, Item2}` coordinates, and
  center the pattern on the board, the same way `LetThereBeLight` does on main.
- Image — resize `cortana.jpg` to the grid and light pixels brighter than a
  luminance threshold, the same idea as the `age` branch.
- Mouse — paint cells directly with a round brush that wraps at the edges.

## Tests

`test_sim.py` checks the rules and the wrap on the CPU backend, with no GPU or
window needed:

```sh
.venv/Scripts/python test_sim.py
```

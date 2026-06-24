# Game of Life on the GPU (Taichi)

A GPU Game of Life that plays like a game. It opens as a four-gun battle: a
Gosper glider gun sits in each corner of the board, all firing toward the
center. The board is a torus — gliders that fly off one edge come back on the
other, asteroids-style — so the four streams cross and tear at the opposing
guns. The gun positions are jittered each game, so no two battles play out the
same. You defend the guns by dropping bombs of life with the mouse. The game
beeps every 15 seconds while a gun still fires, and sounds a knell when the last
one dies.

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

The window opens with the four-gun battle already running. Press space to pause.

Drag any edge to resize the window — the board scales to fill it. The grid stays
a fixed resolution, so resizing zooms the view rather than adding cells, and
drawing still lands on the right cell at any size. Scaling uses nearest-neighbor
sampling, so cells stay crisp squares instead of blurring into blobs.

The display runs at your monitor's refresh, but the simulation is decoupled from
it: the steps/frame slider advances many generations per drawn frame, so the
board can evolve thousands of generations a second while the window stays smooth.
The panel shows live fps and generations/sec.

## Controls

- `space` — run or pause
- left mouse — drop a random bomb of life (one per click)
- right mouse — erase (hold and drag)
- `j` — restart the four-gun battle
- `c` — clear the board
- `r` — reseed at random
- `i` — reseed from `cortana.jpg`
- `g` — scatter gliders flying in random directions
- `k` — place a single Gosper glider gun
- `z` / `x` — shrink or grow the bomb
- `esc` — quit

The panel also has sliders for steps/frame (simulation speed) and bomb size.

## Seeding

- Four guns — the battle places a Gosper glider gun in each corner in code,
  jittered toward the center by a random offset so each game differs.
  `initialState.json` keeps the fixed corner layout as a deterministic
  reference; the game does not load it.
- Bomb — each left click writes a random splat of live cells in a disk.
- Random — light each cell with a fixed probability.
- Image — resize `cortana.jpg` to the grid and light pixels brighter than a
  luminance threshold, the same idea as the `age` branch.
- Gliders — scatter gliders, or place a single Gosper gun.

## Detecting a downed gun

A Gosper gun is a period-30 oscillator, so each gun's corner region cycles
through 30 fixed signatures. The game captures those healthy signatures once per
orientation from an isolated reference gun, then watches each placed gun's
region: one that stops matching its signatures has been destroyed. That count
drives the panel readout and the audio.

## Tests

`test_sim.py` checks the rules and the wrap on the CPU backend, with no GPU or
window needed:

```sh
.venv/Scripts/python test_sim.py
```

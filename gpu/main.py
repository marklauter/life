"""Native display and input for the GPU Game of Life.

It starts as a four-gun battle: a Gosper glider gun sits in each corner of the
torus, all firing toward the center, so their glider streams cross and tear at
the opposing guns. Defend them — each left click drops a random "bomb" of life
to disrupt incoming gliders, and the right button drags to erase.

The window is resizable — drag any edge and the board scales to fill it, keeping
its aspect ratio. The simulation stays a fixed grid; resizing only zooms the
view. Cursor positions map through the fit, so input lands on the right cell.

Controls:
  space   run / pause
  LMB     drop a random bomb of life (one per click)
  RMB     erase (hold and drag)
  j       restart the four-gun battle
  c       clear
  r       reseed random
  i       reseed from cortana.jpg
  g       scatter gliders
  k       place a single Gosper glider gun
  z / x   shrink / grow bomb size
  esc     quit
"""

import time

import taichi as ti

from sim import Simulation

ti.init(arch=ti.gpu)

WIDTH = HEIGHT = 512
CHANCE = 0.2
# vsync=True presents at your monitor's refresh (120 Hz here). Set False to
# uncap the display entirely. Either way, steps/frame decouples the sim from it.
VSYNC = True

sim = Simulation(WIDTH, HEIGHT)
sim.seed_json("initialState.json")

window = ti.ui.Window("life — gpu (taichi)", (WIDTH, HEIGHT), vsync=VSYNC)
canvas = window.get_canvas()
gui = window.get_gui()


# Crisp scaling: upscale the grid with nearest-neighbor sampling into a fitted
# rectangle (dw x dh) offset by (ox, oy), so GGUI never blurs a cell and the
# board keeps its aspect ratio. Pixels outside the rectangle are the letterbox.
@ti.kernel
def upscale(src: ti.template(), dst: ti.template(),
            sw: int, sh: int, dw: int, dh: int, ox: int, oy: int):
    for i, j in dst:
        x = i - ox
        y = j - oy
        v = 0.0
        if 0 <= x < dw and 0 <= y < dh:
            v = ti.cast(src[x * sw // dw, y * sh // dh], ti.f32)
        dst[i, j] = ti.Vector([v, v, v])  # white cells on black


_display_shape = None
_display_tree = None
display_img = None


def ensure_display(w: int, h: int):
    """Allocate the display image at the window size, reusing it until resize."""
    global _display_shape, _display_tree, display_img
    if _display_shape == (w, h):
        return
    if _display_tree is not None:
        _display_tree.destroy()
    builder = ti.FieldsBuilder()
    display_img = ti.Vector.field(3, ti.f32)
    builder.dense(ti.ij, (w, h)).place(display_img)
    _display_tree = builder.finalize()
    _display_shape = (w, h)


running = True
brush = 6
prev_lmb = False
# Decouple the sim from the 60 Hz display: advance many generations per frame.
steps_per_frame = 1

mark_time = time.perf_counter()
mark_gens = 0
mark_frames = 0
frames = 0
fps = 0.0
gens_per_sec = 0.0


def cursor_cell(w: int, h: int, dw: int, dh: int, ox: int, oy: int):
    """Map the cursor through the fitted rectangle to a grid cell, or None."""
    mx, my = window.get_cursor_pos()
    px = mx * w - ox
    py = my * h - oy
    if 0 <= px < dw and 0 <= py < dh:
        return int(px * WIDTH / dw), int(py * HEIGHT / dh)
    return None


while window.running:
    # Fit the square grid into the window at its aspect ratio (letterboxed).
    w, h = window.get_window_shape()
    scale = min(w / WIDTH, h / HEIGHT)
    dw = max(1, int(WIDTH * scale))
    dh = max(1, int(HEIGHT * scale))
    ox = (w - dw) // 2
    oy = (h - dh) // 2

    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            window.running = False
        elif e.key == ti.ui.SPACE:
            running = not running
        elif e.key == "r":
            sim.seed_random(CHANCE)
            running = False
        elif e.key == "j":
            sim.seed_json("initialState.json")  # restart the four-gun battle
            running = True
        elif e.key == "i":
            sim.seed_image("cortana.jpg")
            running = False
        elif e.key == "c":
            sim.clear()
            running = False
        elif e.key == "g":
            sim.scatter_gliders(40)
            running = True
        elif e.key == "k":
            sim.seed_gun()
            running = True
        elif e.key == "z":
            brush = max(1, brush - 1)
        elif e.key == "x":
            brush += 1

    # Each left click drops a random "bomb" of life; rising edge = one per click.
    lmb = window.is_pressed(ti.ui.LMB)
    if lmb and not prev_lmb:
        cell = cursor_cell(w, h, dw, dh, ox, oy)
        if cell is not None:
            sim.bomb(cell[0], cell[1], brush)
    prev_lmb = lmb

    # Right button drags to erase, for defending the guns.
    if window.is_pressed(ti.ui.RMB):
        cell = cursor_cell(w, h, dw, dh, ox, oy)
        if cell is not None:
            sim.paint(cell[0], cell[1], brush, 0)

    if running:
        for _ in range(steps_per_frame):
            sim.step()

    # Measure real fps and generations/sec a couple of times a second.
    frames += 1
    now = time.perf_counter()
    if now - mark_time >= 0.5:
        dt = now - mark_time
        fps = (frames - mark_frames) / dt
        gens_per_sec = (sim.generations - mark_gens) / dt
        mark_time, mark_frames, mark_gens = now, frames, sim.generations

    ensure_display(w, h)
    upscale(sim.a, display_img, WIDTH, HEIGHT, dw, dh, ox, oy)
    canvas.set_image(display_img)
    with gui.sub_window("life", 0.0, 0.0, 0.46, 0.24):
        gui.text(f"{'running' if running else 'paused'}   gen {sim.generations}")
        gui.text(f"{fps:.0f} fps   {gens_per_sec:,.0f} gen/s")
        steps_per_frame = gui.slider_int("steps/frame", steps_per_frame, 1, 500)
        brush = gui.slider_int("bomb size", brush, 1, 40)
        gui.text("[LMB] bomb  [RMB] erase")
        gui.text("[j] four-gun battle  [c]lear")
        gui.text("[r]andom [i]mage [g]liders [k]gun")
    window.show()

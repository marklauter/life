"""Native display and input for the GPU Game of Life.

Run it, then draw: hold the left mouse button and drag to paint live cells,
right button to erase. The board is a torus, so brush strokes and gliders wrap
around the edges. Seed it the way main does (centered JSON pattern) by default,
or reseed at runtime from the keyboard.

The window is resizable — drag any edge and the board scales to fill it. The
simulation stays a fixed grid; resizing only zooms the view. Cursor positions
are normalized, so drawing lands on the right cell at any window size.

Controls:
  space   run / pause
  LMB     draw   (hold and drag)
  RMB     erase  (hold and drag)
  r       reseed random
  j       reseed from initialState.json
  i       reseed from cortana.jpg
  c       clear
  z / x   shrink / grow brush
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


# Crisp scaling: upscale the grid to the window's exact pixel size with
# nearest-neighbor sampling, so GGUI displays it 1:1 and never blurs a cell.
@ti.kernel
def upscale(src: ti.template(), dst: ti.template(), sw: int, sh: int, dw: int, dh: int):
    for i, j in dst:
        v = ti.cast(src[i * sw // dw, j * sh // dh], ti.f32)
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


running = False
brush = 6
# Decouple the sim from the 60 Hz display: advance many generations per frame.
steps_per_frame = 1

mark_time = time.perf_counter()
mark_gens = 0
mark_frames = 0
frames = 0
fps = 0.0
gens_per_sec = 0.0


def paint_at(value: int):
    mx, my = window.get_cursor_pos()
    sim.paint(int(mx * WIDTH), int(my * HEIGHT), brush, value)


while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            window.running = False
        elif e.key == ti.ui.SPACE:
            running = not running
        elif e.key == "r":
            sim.seed_random(CHANCE)
            running = False
        elif e.key == "j":
            sim.seed_json("initialState.json")
            running = False
        elif e.key == "i":
            sim.seed_image("cortana.jpg")
            running = False
        elif e.key == "c":
            sim.clear()
            running = False
        elif e.key == "z":
            brush = max(1, brush - 1)
        elif e.key == "x":
            brush += 1

    if window.is_pressed(ti.ui.LMB):
        paint_at(1)
    if window.is_pressed(ti.ui.RMB):
        paint_at(0)

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

    w, h = window.get_window_shape()
    ensure_display(w, h)
    upscale(sim.a, display_img, WIDTH, HEIGHT, w, h)
    canvas.set_image(display_img)
    with gui.sub_window("life", 0.0, 0.0, 0.46, 0.24):
        gui.text(f"{'running' if running else 'paused'}   gen {sim.generations}")
        gui.text(f"{fps:.0f} fps   {gens_per_sec:,.0f} gen/s")
        steps_per_frame = gui.slider_int("steps/frame", steps_per_frame, 1, 500)
        brush = gui.slider_int("brush", brush, 1, 40)
        gui.text("[LMB] draw  [RMB] erase")
        gui.text("[r]andom [j]son [i]mage [c]lear")
    window.show()

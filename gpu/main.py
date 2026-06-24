"""Native display and input for the GPU Game of Life — the four-gun battle.

It opens as a battle: a Gosper glider gun sits in each corner of the torus, all
firing toward the center, so their glider streams cross and tear at the opposing
guns. The gun positions are jittered each game, so no two battles play out the
same. Defend them — each left click drops a random "bomb" of life, and the right
button drags to erase. A triumphant chime sounds every 10 seconds while a gun
still fires, a sad note when any gun falls, and a knell when the last one dies.

The window is resizable and keeps the board's aspect ratio; scaling is
nearest-neighbor, so cells stay crisp squares. The simulation runs independently
of the display — the steps/frame slider advances many generations per drawn frame.

Controls:
  space   run / pause
  LMB     drop a random bomb of life (one per click)
  RMB     erase (hold and drag)
  j       restart the four-gun battle (new random layout)
  a       launch the asteroids ship over a glider field
  w/a/d   fly the ship: thrust / turn left / turn right
  c       clear
  r       reseed random
  i       reseed from cortana.jpg
  g       scatter gliders
  k       place a single Gosper glider gun
  z / x   shrink / grow bomb size
  esc     quit
"""

import threading
import time

import numpy as np
import taichi as ti

from sim import GOSPER_GUN, Simulation
from ship import Ship

try:
    import winsound

    def _beep(tones):
        for freq, ms in tones:
            winsound.Beep(freq, ms)

    def play(tones):
        """Play (freq_hz, ms) beeps on a background thread, never blocking."""
        threading.Thread(target=_beep, args=(tones,), daemon=True).start()

except ImportError:  # non-Windows: run silently

    def play(tones):
        pass


TRIUMPH = [(523, 90), (659, 90), (784, 90), (1047, 200)]  # a gun still fires (every 10s)
GUN_DOWN = [(440, 130), (330, 280)]  # one gun died (others remain)
DEATH_KNELL = [(392, 220), (330, 240), (262, 280), (196, 700)]  # the last gun died

ti.init(arch=ti.gpu)

WIDTH = HEIGHT = 512
CHANCE = 0.2
MARGIN = 2          # corner inset for the guns
JITTER = 30         # max random offset of each gun toward the center
SETTLE = 120        # generations before a gun's box reaches its steady cycle
# vsync=True presents at your monitor's refresh (120 Hz here). Set False to
# uncap the display entirely. Either way, steps/frame decouples the sim from it.
VSYNC = True

sim = Simulation(WIDTH, HEIGHT)
ship = Ship(WIDTH, HEIGHT)
# Vector ship outline: 4 segments = 8 endpoints. A root field, so it is declared
# before the first kernel launch (the phase capture below).
ship_verts = ti.Vector.field(2, ti.f32, shape=8)

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


# Gun liveness: a Gosper gun is a period-30 oscillator. Capture the healthy
# phase signatures of each orientation once, from an isolated reference gun;
# a placed gun whose box stops matching its orientation's phases has died.
def gun_box(ax: int, ay: int, fx: int, fy: int, pad: int = 2):
    xs = [ax + fx * dx for dx, _ in GOSPER_GUN]
    ys = [ay + fy * dy for _, dy in GOSPER_GUN]
    return (min(xs) - pad, min(ys) - pad,
            max(xs) - min(xs) + 1 + 2 * pad, max(ys) - min(ys) + 1 + 2 * pad)


def box_sig(arr, box):
    x0, y0, w, h = box
    xs = np.arange(x0, x0 + w) % arr.shape[0]
    ys = np.arange(y0, y0 + h) % arr.shape[1]
    return arr[np.ix_(xs, ys)].astype(np.uint8).tobytes()


def orientation_phases(fx: int, fy: int, period: int = 30):
    """Steady-state box signatures of a lone gun in one mirror orientation."""
    saved, saved_gen = sim.a.to_numpy().copy(), sim.generations
    sim.clear()
    ax, ay = WIDTH // 2, HEIGHT // 2
    sim.stamp(GOSPER_GUN, ax, ay, fx, fy)
    box = gun_box(ax, ay, fx, fy)
    for _ in range(SETTLE):
        sim.step()
    phases = set()
    for _ in range(period):
        phases.add(box_sig(sim.a.to_numpy(), box))
        sim.step()
    sim.a.from_numpy(saved)
    sim.generations = saved_gen
    return phases


ORIENTATIONS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
ORIENTATION_PHASES = {o: orientation_phases(*o) for o in ORIENTATIONS}


def cursor_cell(w: int, h: int, dw: int, dh: int, ox: int, oy: int):
    """Map the cursor through the fitted rectangle to a grid cell, or None."""
    mx, my = window.get_cursor_pos()
    px = mx * w - ox
    py = my * h - oy
    if 0 <= px < dw and 0 <= py < dh:
        return int(px * WIDTH / dw), int(py * HEIGHT / dh)
    return None


boxes = []
phase_sets = []
alive = []
last_beep = time.perf_counter()


def start_battle():
    """Clear and place a jittered gun in each corner, all firing toward center."""
    global boxes, phase_sets, alive, last_beep
    sim.clear()
    corners = [
        (MARGIN, MARGIN, 1, 1),                              # bottom-left
        (WIDTH - 1 - MARGIN, MARGIN, -1, 1),                 # bottom-right
        (MARGIN, HEIGHT - 1 - MARGIN, 1, -1),                # top-left
        (WIDTH - 1 - MARGIN, HEIGHT - 1 - MARGIN, -1, -1),   # top-right
    ]
    boxes, phase_sets = [], []
    for cx, cy, fx, fy in corners:
        ax = cx + fx * int(np.random.randint(0, JITTER + 1))  # jitter inward
        ay = cy + fy * int(np.random.randint(0, JITTER + 1))
        sim.stamp(GOSPER_GUN, ax, ay, fx, fy)
        boxes.append(gun_box(ax, ay, fx, fy))
        phase_sets.append(ORIENTATION_PHASES[(fx, fy)])
    alive = [True] * len(boxes)
    last_beep = time.perf_counter()


running = True
battle = True
ship_mode = False
brush = 6
prev_lmb = False
steps_per_frame = 1
last_check = 0.0
prev_time = time.perf_counter()

mark_time = time.perf_counter()
mark_gens = 0
mark_frames = 0
frames = 0
fps = 0.0
gens_per_sec = 0.0

start_battle()

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
        elif e.key == "j":
            start_battle()
            battle, ship_mode, running = True, False, True
        elif e.key == "c":
            sim.clear()
            battle, ship_mode, running = False, False, False
        elif e.key == "r":
            sim.seed_random(CHANCE)
            battle, ship_mode, running = False, False, False
        elif e.key == "i":
            sim.seed_image("cortana.jpg")
            battle, ship_mode, running = False, False, False
        elif e.key == "g":
            sim.scatter_gliders(40)
            battle, ship_mode, running = False, False, True
        elif e.key == "k":
            sim.seed_gun()
            battle, ship_mode, running = False, False, True
        elif e.key == "a" and not ship_mode:
            # First 'a' launches the asteroids ship over a glider field; held 'a'
            # afterward steers it (handled below), so it won't relaunch.
            sim.scatter_gliders(40)
            ship.reset()
            battle, ship_mode, running = False, True, True
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

    # Fly the ship at frame rate, independent of the sim's steps/frame.
    dt = min(now - prev_time, 0.05)
    prev_time = now
    if ship_mode:
        ship.update(dt, window.is_pressed("w"), window.is_pressed("a"),
                    window.is_pressed("d"))

    # Gun liveness and audio. Wait for the boxes to reach their steady cycle, and
    # a gun stays dead once detected.
    if battle and sim.generations >= SETTLE and now - last_check >= 0.25:
        last_check = now
        arr = sim.a.to_numpy()
        before = sum(alive)
        for k, box in enumerate(boxes):
            if alive[k] and box_sig(arr, box) not in phase_sets[k]:
                alive[k] = False
        after = sum(alive)
        if before > after:
            play(DEATH_KNELL if after == 0 else GUN_DOWN)  # knell only for the last
        elif after > 0 and now - last_beep >= 10:
            play(TRIUMPH)
            last_beep = now

    ensure_display(w, h)
    upscale(sim.a, display_img, WIDTH, HEIGHT, dw, dh, ox, oy)
    canvas.set_image(display_img)

    # Draw the vector ship over the board (as a closed outline of line segments).
    if ship_mode:
        nose, right, notch, left = ship.outline()
        pts = [nose, right, right, notch, notch, left, left, nose]
        verts = np.empty((8, 2), dtype=np.float32)
        for idx, (gx, gy) in enumerate(pts):
            verts[idx, 0] = (ox + gx * dw / WIDTH) / w
            verts[idx, 1] = (oy + gy * dh / HEIGHT) / h
        ship_verts.from_numpy(verts)
        canvas.lines(ship_verts, width=0.004, color=(0.3, 1.0, 0.4))

    with gui.sub_window("life", 0.27, 0.0, 0.46, 0.34):  # top-center, clears the corners
        gui.text(f"{'running' if running else 'paused'}   gen {sim.generations}")
        gui.text(f"{fps:.0f} fps   {gens_per_sec:,.0f} gen/s")
        if battle:
            gui.text(f"guns alive: {sum(alive)}/4")
        if ship_mode:
            gui.text("[W] thrust  [A/D] turn")
        steps_per_frame = gui.slider_int("steps/frame", steps_per_frame, 1, 500)
        brush = gui.slider_int("bomb size", brush, 1, 40)
        gui.text("[LMB] bomb  [RMB] erase")
        gui.text("[j] battle  [a]steroids  [c]lear")
        gui.text("[r]andom [i]mage [g]liders [k]gun")
    window.show()

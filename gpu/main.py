"""Native display and input for the GPU Game of Life — the four-gun battle.

It opens as a battle: a Gosper glider gun sits in each corner of the torus, all
firing toward the center, so their glider streams cross and tear at the opposing
guns. The gun positions are jittered each game, so no two battles play out the
same. Defend them — each left click drops a random "bomb" of life, and the right
button drags to erase. A triumphant chime sounds every 10 seconds while a gun
still fires, a sad note when any gun falls, and a knell when the last one dies.

Press '1' for the asteroids ship: fly it over a glider field with A/D to turn
and space to thrust, and the up arrow fires bullets that bomb a splat of new
life where they hit, stirring the board rather than clearing it. Fly into a live
cell and the ship crashes.

The window is resizable and keeps the board's aspect ratio; scaling is
nearest-neighbor, so cells stay crisp squares. The simulation runs independently
of the display — the steps/frame slider advances many generations per drawn frame.

Modes (number keys):
  1       asteroids ship over a glider field
  2       four-gun battle (new random layout)
  3       scatter gliders
  4       single Gosper glider gun
  5       random soup
  6       image (cortana.jpg)
  7       clear

Controls:
  space   run / pause  (in ship mode: thrust)
  a / d   turn the ship left / right
  up      fire bullets (ship mode)
  LMB     drop a random bomb of life (one per click)
  RMB     erase (hold and drag)
  z / x   shrink / grow bomb size
  esc     quit
"""

import math
import os
import tempfile
import threading
import time
import wave

import numpy as np
import taichi as ti

from sim import GOSPER_GUN, Simulation
from ship import Ship

# A short white-noise clip for the thrust rumble, looped while thrusting.
NOISE_WAV = os.path.join(tempfile.gettempdir(), "life_thrust.wav")


def _write_noise(path, seconds=0.4, rate=22050, volume=0.2):
    samples = (np.random.uniform(-1, 1, int(seconds * rate)) * volume * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


_write_noise(NOISE_WAV)

try:
    import winsound

    def _beep(tones):
        for freq, ms in tones:
            winsound.Beep(freq, ms)

    def play(tones):
        """Play (freq_hz, ms) beeps on a background thread, never blocking."""
        threading.Thread(target=_beep, args=(tones,), daemon=True).start()

    def thrust_start():
        winsound.PlaySound(NOISE_WAV, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)

    def thrust_stop():
        winsound.PlaySound(None, winsound.SND_PURGE)

except ImportError:  # non-Windows: run silently

    def play(tones):
        pass

    def thrust_start():
        pass

    def thrust_stop():
        pass


TRIUMPH = [(523, 90), (659, 90), (784, 90), (1047, 200)]  # a gun still fires (every 10s)
GUN_DOWN = [(440, 130), (330, 280)]  # one gun died (others remain)
DEATH_KNELL = [(392, 220), (330, 240), (262, 280), (196, 700)]  # the last gun died
CRASH = [(196, 70), (147, 90), (98, 180)]  # the ship hit a live cell
PEW = [(1200, 30), (700, 35)]  # bullet fired
BOOM = [(330, 30), (160, 70)]  # a bullet blew up a cluster

ti.init(arch=ti.gpu)

WIDTH = HEIGHT = 512
CHANCE = 0.2
MARGIN = 2            # corner inset for the guns
JITTER = 30          # max random offset of each gun toward the center
SETTLE = 120         # generations before a gun's box reaches its steady cycle
COLLIDE_R = 6        # ship collision radius in cells
SPAWN_SAFE = 1.5     # seconds of invulnerability after the ship respawns
DEATH_TIME = 1.2     # seconds the ship spends broken apart before respawning
BULLET_SPEED = 420.0  # cells per second
BULLET_LIFE = 1.0    # seconds before a bullet expires
MAX_BULLETS = 64
BULLET_HIT_R = 3     # bullet hitbox radius (cells), so it catches sparse gliders
BLAST_R = 4          # cells cleared in the blast when a bullet hits
SHIP_COLOR = (1.0, 1.0, 1.0)  # same white as the life cells
# vsync=True presents at your monitor's refresh (120 Hz here). Set False to
# uncap the display entirely. Either way, steps/frame decouples the sim from it.
VSYNC = True


def _disk_offsets(r):
    """Offset arrays (dx, dy) of the cells within radius r — for vectorized hits."""
    pairs = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
             if dx * dx + dy * dy <= r * r]
    a = np.array(pairs, dtype=np.int64)
    return a[:, 0], a[:, 1]


HIT_DX, HIT_DY = _disk_offsets(BULLET_HIT_R)
BLAST_DX, BLAST_DY = _disk_offsets(BLAST_R)

sim = Simulation(WIDTH, HEIGHT)
ship = Ship(WIDTH, HEIGHT)
# Overlay vertex buffers — root fields, declared before the first kernel launch
# (the phase capture below). Ship outline is 4 segments (8 endpoints), flame is
# 2 segments (4 endpoints).
ship_verts = ti.Vector.field(2, ti.f32, shape=8)
ship_flame = ti.Vector.field(2, ti.f32, shape=4)
bullet_pos = ti.Vector.field(2, ti.f32, shape=MAX_BULLETS)
debris_seg = ti.Vector.field(2, ti.f32, shape=2)  # one fragment at a time

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


@ti.kernel
def cells_in_disk(src: ti.template(), cx: int, cy: int, r: int) -> ti.i32:
    """Count live cells within radius r of (cx, cy) — the ship collision test."""
    count = 0
    for _ in range(1):  # serialize the small scan; no parallel reduction
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    count += src[(cx + dx) % WIDTH, (cy + dy) % HEIGHT]
    return count


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


def to_ndc(gx, gy, w, h, dw, dh, ox, oy):
    """Map a grid point through the aspect-fit to normalized window coords."""
    return (ox + gx * dw / WIDTH) / w, (oy + gy * dh / HEIGHT) / h


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
ship_alive = True
ship_dead_until = 0.0
ship_hits = 0
ship_safe_until = 0.0
bullets = []
debris = []
brush = 6
prev_lmb = False
thrust_sound = False
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
            if not ship_mode:  # in ship mode, space is thrust (held, handled below)
                running = not running
        elif e.key == ti.ui.UP and ship_mode:  # fire a pixel bullet from the nose
            if len(bullets) < MAX_BULLETS:
                nx, ny = ship.outline()[0]
                vx = math.cos(ship.angle) * BULLET_SPEED + ship.vx
                vy = math.sin(ship.angle) * BULLET_SPEED + ship.vy
                bullets.append([nx, ny, vx, vy, 0.0])
                play(PEW)
        elif e.key == "1":
            # Asteroids ship over a glider field; turn with A/D, thrust with space.
            sim.scatter_gliders(40)
            ship.reset()
            ship_hits, bullets, debris = 0, [], []
            ship_alive, ship_dead_until = True, 0.0
            ship_safe_until = time.perf_counter() + SPAWN_SAFE
            battle, ship_mode, running = False, True, True
        elif e.key == "2":
            start_battle()
            battle, ship_mode, running, bullets = True, False, True, []
        elif e.key == "3":
            sim.scatter_gliders(40)
            battle, ship_mode, running, bullets = False, False, True, []
        elif e.key == "4":
            sim.seed_gun()
            battle, ship_mode, running, bullets = False, False, True, []
        elif e.key == "5":
            sim.seed_random(CHANCE)
            battle, ship_mode, running, bullets = False, False, False, []
        elif e.key == "6":
            sim.seed_image("cortana.jpg")
            battle, ship_mode, running, bullets = False, False, False, []
        elif e.key == "7":
            sim.clear()
            battle, ship_mode, running, bullets = False, False, False, []
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
        elapsed = now - mark_time
        fps = (frames - mark_frames) / elapsed
        gens_per_sec = (sim.generations - mark_gens) / elapsed
        mark_time, mark_frames, mark_gens = now, frames, sim.generations

    # Fly the ship at frame rate, independent of the sim's steps/frame.
    dt = min(now - prev_time, 0.05)
    prev_time = now
    thrust_on = ship_mode and ship_alive and window.is_pressed(ti.ui.SPACE)
    if thrust_on and not thrust_sound:
        thrust_start()
        thrust_sound = True
    elif not thrust_on and thrust_sound:
        thrust_stop()
        thrust_sound = False
    if ship_mode and ship_alive:
        ship.update(dt, thrust_on, window.is_pressed("a"), window.is_pressed("d"))
        if now > ship_safe_until and cells_in_disk(sim.a, int(ship.x), int(ship.y), COLLIDE_R) > 0:
            play(CRASH)
            ship_hits += 1
            ship_alive = False
            ship_dead_until = now + DEATH_TIME
            # Break the outline into drifting, spinning fragments.
            pts = ship.outline()
            debris = []
            for a, b in [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[0])]:
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                hx, hy = (b[0] - a[0]) / 2, (b[1] - a[1]) / 2
                outx, outy = mx - ship.x, my - ship.y
                norm = math.hypot(outx, outy) or 1.0
                push = np.random.uniform(20, 50)
                vx = ship.vx + outx / norm * push + np.random.uniform(-10, 10)
                vy = ship.vy + outy / norm * push + np.random.uniform(-10, 10)
                debris.append([mx, my, hx, hy, vx, vy, np.random.uniform(-2.5, 2.5), 0.0])
    elif ship_mode and now >= ship_dead_until:
        ship.reset()
        ship_alive = True
        ship_safe_until = now + SPAWN_SAFE
        debris = []

    # Drift and spin the debris fragments while the ship is broken apart.
    if ship_mode and debris:
        survivors = []
        for f in debris:
            ca, sa = math.cos(f[6] * dt), math.sin(f[6] * dt)
            f[2], f[3] = f[2] * ca - f[3] * sa, f[2] * sa + f[3] * ca
            f[0] = (f[0] + f[4] * dt) % WIDTH
            f[1] = (f[1] + f[5] * dt) % HEIGHT
            f[7] += dt
            if f[7] < DEATH_TIME:
                survivors.append(f)
        debris = survivors

    # Move bullets; clear any life cell they hit, expire the rest.
    if ship_mode and bullets:
        arr = sim.a.to_numpy()
        modified = False
        survivors = []
        for b in bullets:
            b[0] = (b[0] + b[2] * dt) % WIDTH
            b[1] = (b[1] + b[3] * dt) % HEIGHT
            b[4] += dt
            bx, by = int(b[0]), int(b[1])
            if arr[(bx + HIT_DX) % WIDTH, (by + HIT_DY) % HEIGHT].any():
                # Bomb a random splat of life like a mouse click, so the board
                # keeps evolving — bullets stir it up, they never blank it.
                mask = np.random.random(BLAST_DX.shape[0]) < 0.5
                arr[(bx + BLAST_DX[mask]) % WIDTH, (by + BLAST_DY[mask]) % HEIGHT] = 1
                modified = True
            elif b[4] < BULLET_LIFE:
                survivors.append(b)
        bullets = survivors
        if modified:
            sim.a.from_numpy(np.ascontiguousarray(arr, dtype=np.int32))
            play(BOOM)  # a bullet blew up a cluster this frame

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

    line_w = 1.0 / h  # one pixel — as thin as the line renderer goes
    if ship_mode and ship_alive:
        # Ship outline as a closed loop of line segments.
        nose, right, notch, left = ship.outline()
        pts = [nose, right, right, notch, notch, left, left, nose]
        verts = np.empty((8, 2), dtype=np.float32)
        for idx, (gx, gy) in enumerate(pts):
            verts[idx] = to_ndc(gx, gy, w, h, dw, dh, ox, oy)
        ship_verts.from_numpy(verts)
        canvas.lines(ship_verts, width=line_w, color=SHIP_COLOR)

        # Exhaust flame while thrusting (flicker for the classic look).
        if thrust_on and (frames // 3) % 2 == 0:
            fl, ft, fr = ship.flame()
            fpts = [fl, ft, ft, fr]
            fverts = np.empty((4, 2), dtype=np.float32)
            for idx, (gx, gy) in enumerate(fpts):
                fverts[idx] = to_ndc(gx, gy, w, h, dw, dh, ox, oy)
            ship_flame.from_numpy(fverts)
            canvas.lines(ship_flame, width=line_w, color=SHIP_COLOR)

    # Broken-apart fragments, each fading as it ages.
    if ship_mode and debris:
        for f in debris:
            seg = np.empty((2, 2), dtype=np.float32)
            seg[0] = to_ndc(f[0] - f[2], f[1] - f[3], w, h, dw, dh, ox, oy)
            seg[1] = to_ndc(f[0] + f[2], f[1] + f[3], w, h, dw, dh, ox, oy)
            debris_seg.from_numpy(seg)
            fade = max(0.0, 1.0 - f[7] / DEATH_TIME)
            canvas.lines(debris_seg, width=line_w, color=(fade, fade, fade))

    # Pixel bullets keep flying even while the ship is broken apart.
    if ship_mode and bullets:
        bnp = np.full((MAX_BULLETS, 2), -1.0, dtype=np.float32)
        for i, b in enumerate(bullets[:MAX_BULLETS]):
            bnp[i] = to_ndc(b[0], b[1], w, h, dw, dh, ox, oy)
        bullet_pos.from_numpy(bnp)
        canvas.circles(bullet_pos, radius=2.0 / h, color=SHIP_COLOR)

    with gui.sub_window("life", 0.27, 0.0, 0.46, 0.4):  # top-center, clears the corners
        gui.text(f"{'running' if running else 'paused'}   gen {sim.generations}")
        gui.text(f"{fps:.0f} fps   {gens_per_sec:,.0f} gen/s")
        if battle:
            gui.text(f"guns alive: {sum(alive)}/4")
        if ship_mode:
            gui.text(f"crashes {ship_hits}")
            gui.text("[space] thrust  [A/D] turn  [up] fire")
        steps_per_frame = gui.slider_int("steps/frame", steps_per_frame, 1, 500)
        brush = gui.slider_int("bomb size", brush, 1, 40)
        gui.text("[LMB] bomb  [RMB] erase")
        gui.text("1 asteroids  2 battle  3 gliders  4 gun")
        gui.text("5 random  6 image  7 clear")
    window.show()

thrust_stop()  # silence the rumble on exit

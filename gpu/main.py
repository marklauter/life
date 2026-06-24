"""Native display and input for the GPU Game of Life.

Run it, then draw: hold the left mouse button and drag to paint live cells,
right button to erase. The board is a torus, so brush strokes and gliders wrap
around the edges. Seed it the way main does (centered JSON pattern) by default,
or reseed at runtime from the keyboard.

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

import taichi as ti

from sim import Simulation

ti.init(arch=ti.gpu)

WIDTH = HEIGHT = 512
CHANCE = 0.2

sim = Simulation(WIDTH, HEIGHT)
sim.seed_json("initialState.json")

gui = ti.GUI("life — gpu (taichi)", res=(WIDTH, HEIGHT))
running = False
brush = 6


def paint_at(value: int):
    mx, my = gui.get_cursor_pos()
    sim.paint(int(mx * WIDTH), int(my * HEIGHT), brush, value)


while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == ti.GUI.SPACE:
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

    if gui.is_pressed(ti.GUI.LMB):
        paint_at(1)
    if gui.is_pressed(ti.GUI.RMB):
        paint_at(0)

    if running:
        sim.step()

    gui.set_image(sim.render())
    gui.text(
        content=f"{'running' if running else 'paused'}   "
        f"gen {sim.generations}   brush {brush}",
        pos=(0.01, 0.99),
        color=0x00FF00,
    )
    gui.text(
        content="[space] run  [LMB] draw  [RMB] erase  "
        "[r]andom [j]son [i]mage [c]lear  [z/x] brush",
        pos=(0.01, 0.04),
        color=0x888888,
    )
    gui.show()

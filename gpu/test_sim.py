"""Headless correctness checks for the simulation — no GPU or display needed.

Run with: python test_sim.py
Uses the CPU backend so it works anywhere, including CI.
"""

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

from sim import Simulation  # noqa: E402  (init must precede field allocation)


def test_blinker_oscillates():
    """A blinker flips between vertical and horizontal with period 2."""
    sim = Simulation(8, 8)
    start = np.zeros((8, 8), dtype=np.int32)
    start[4, 3] = start[4, 4] = start[4, 5] = 1  # vertical bar
    sim.a.from_numpy(start)

    sim.step()
    g1 = sim.a.to_numpy()
    assert g1[3, 4] == g1[4, 4] == g1[5, 4] == 1, "should rotate to horizontal"
    assert g1[4, 3] == g1[4, 5] == 0, "vertical ends should die"

    sim.step()
    assert (sim.a.to_numpy() == start).all(), "period 2: back to vertical"
    print("blinker oscillation OK")


def test_toroidal_wrap():
    """A blinker straddling the top/bottom edge rotates across the wrap."""
    sim = Simulation(5, 5)
    start = np.zeros((5, 5), dtype=np.int32)
    start[0, 4] = start[0, 0] = start[0, 1] = 1  # vertical bar wrapping row 4->0->1
    sim.a.from_numpy(start)

    sim.step()
    g = sim.a.to_numpy()
    # Rotates to a horizontal bar through column 4->0->1 on row 0, using the wrap.
    assert g[4, 0] == g[0, 0] == g[1, 0] == 1, f"wrap rotation failed:\n{g.T}"
    print("toroidal wrap OK")


def test_paint_wraps():
    """A brush at the corner paints across the wrapped edges."""
    sim = Simulation(20, 20)
    sim.clear()
    sim.paint(0, 0, 3, 1)  # circle centered on the corner
    g = sim.a.to_numpy()
    assert g[0, 0] == 1, "brush center should be set"
    assert g[19, 0] == 1 and g[0, 19] == 1, "brush should wrap to far edges"
    print("paint wrap OK")


def test_seed_json_centers():
    """JSON seeding lights cells and centers them on the board."""
    sim = Simulation(128, 128)
    sim.seed_json("initialState.json")
    assert sim.a.to_numpy().sum() > 0, "json seed should light some cells"
    print("json seed OK")


if __name__ == "__main__":
    test_blinker_oscillates()
    test_toroidal_wrap()
    test_paint_wraps()
    test_seed_json_centers()
    print("all tests passed")

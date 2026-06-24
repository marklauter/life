"""A 1980s-Asteroids player ship — pure Python, no GPU or display.

Newtonian flight on a torus: thrust accelerates along the heading, momentum
carries between frames, a light drag bleeds off speed, and position wraps at the
edges. Kept separate from the simulation and renderer so the physics is testable.
"""

import math


class Ship:
    def __init__(self, width, height, size=12.0, turn_rate=3.2,
                 thrust=240.0, friction=0.2, max_speed=280.0):
        self.w = width
        self.h = height
        self.size = size          # nose-to-tail length in grid cells
        self.turn_rate = turn_rate  # radians per second
        self.thrust = thrust        # acceleration, cells per second squared
        self.friction = friction    # velocity damping per second
        self.max_speed = max_speed  # cells per second
        self.reset()

    def reset(self):
        self.x = self.w / 2
        self.y = self.h / 2
        self.vx = 0.0
        self.vy = 0.0
        self.angle = math.pi / 2  # pointing up

    def update(self, dt, thrust, left, right):
        """Advance one frame. thrust/left/right are booleans (W / A / D held)."""
        if left:
            self.angle += self.turn_rate * dt
        if right:
            self.angle -= self.turn_rate * dt
        if thrust:
            self.vx += math.cos(self.angle) * self.thrust * dt
            self.vy += math.sin(self.angle) * self.thrust * dt
        damp = max(0.0, 1.0 - self.friction * dt)
        self.vx *= damp
        self.vy *= damp
        speed = math.hypot(self.vx, self.vy)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale
        self.x = (self.x + self.vx * dt) % self.w
        self.y = (self.y + self.vy * dt) % self.h

    def outline(self):
        """The vector ship as four points: nose, right rear, tail notch, left rear."""
        c, s = math.cos(self.angle), math.sin(self.angle)
        local = [
            (self.size, 0.0),                       # nose
            (-self.size * 0.7, self.size * 0.6),    # right rear
            (-self.size * 0.4, 0.0),                # tail notch
            (-self.size * 0.7, -self.size * 0.6),   # left rear
        ]
        return [(self.x + lx * c - ly * s, self.y + lx * s + ly * c) for lx, ly in local]

    def flame(self):
        """Exhaust flame chevron behind the tail: left base, tip, right base."""
        c, s = math.cos(self.angle), math.sin(self.angle)
        local = [
            (-self.size * 0.4, self.size * 0.28),   # left base at the notch
            (-self.size * 1.05, 0.0),               # flame tip behind the ship
            (-self.size * 0.4, -self.size * 0.28),  # right base at the notch
        ]
        return [(self.x + lx * c - ly * s, self.y + lx * s + ly * c) for lx, ly in local]

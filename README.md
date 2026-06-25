# life

Conway's Game of Life in C#, and a workbench for variations. Each branch takes the same core and pushes on a different idea — faster neighbor counting, new rules, or different hardware.

https://en.wikipedia.org/wiki/Conway's_Game_of_Life

## Branches

- [main](https://github.com/marklauter/life/tree/main) — the mainline: two simulation engines, JSON seeding, and rendering straight to the form.
- [better-neighbors](https://github.com/marklauter/life/tree/better-neighbors) — performance: count neighbors less often by tracking only the live cells (merged here as pull request #3).
- [age](https://github.com/marklauter/life/tree/age) — cell senescence with stochastic survival, then seeding the grid from an image.
- [gpu-taichi](https://github.com/marklauter/life/tree/gpu-taichi) — a Python/Taichi reimplementation that runs the simulation on the GPU and grows into a game: a four-gun glider battle and a flyable Asteroids ship.

The [wiki](https://github.com/marklauter/life/wiki) describes what each branch explores.

namespace GameOfLife
{
    internal enum PixelState
    {
        Dead = 0,
        Alive = 1,
        Dying = 2,
    }

    internal class Simulation
    {
        public long Generations { get; private set; } = 0;
        public event EventHandler<EventArgs>? SimulationChanged;
        private readonly Bitmap image;

        private readonly PixelState[,,] cells;
        private int primary = 0;
        private readonly int width;
        private readonly int height;

        public Simulation(int width, int height, int chance)
        {
            this.cells = new PixelState[2, width, height];
            this.image = new Bitmap(width, height);
            this.width = width;
            this.height = height;
            this.Randomize(chance);
        }

        private void Randomize(int chance)
        {
            var rng = new Random(DateTime.UtcNow.Millisecond);
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    this.cells[this.primary, x, y] = rng.Next(20) <= 20 * (chance / 100.0)
                        ? PixelState.Alive
                        : PixelState.Dead;
                }
            }
        }

        private async void RunAsync()
        {
            await Task.Run(async () =>
            {
                await Task.Delay(50);
                this.ApplyRules();
                this.RenderBitmap();
                ++this.Generations;
            });

            this.SimulationChanged?.Invoke(this, EventArgs.Empty);
        }

        private void RenderBitmap()
        {
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    this.image.SetPixel(x, y, GetColor(this.cells[this.primary, x, y]));
                }
            }
        }

        private void ApplyRules()
        {
            var successor = this.primary ^ 1;
            // https://en.wikipedia.org/wiki/Conway's_Game_of_Life
            var rng = new Random(DateTime.UtcNow.Millisecond);
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    var neighbors = this.CountLivingNeighbors(x, y);

                    this.cells[this.primary, x, y] = this.cells[this.primary, x, y] == PixelState.Dying
                        ? PixelState.Dead
                        : this.cells[this.primary, x, y];

                    // the RNG creates a chance that a cell will not die. this causes oscillators to sometimes spawn into puffers and gliders
                    this.cells[successor, x, y] = this.cells[this.primary, x, y] switch
                    {
                        PixelState.Dead => neighbors == 3 ? PixelState.Alive : PixelState.Dead,
                        _ => neighbors == 2 || neighbors == 3 || rng.NextDouble() < 0.0005 ? PixelState.Alive : PixelState.Dying,
                    };
                }
            }

            this.primary = successor;
        }

        private int CountLivingNeighbors(int x, int y)
        {
            var xmin = x - 1;
            xmin = xmin != -1 ? xmin : this.width - 1; // wrap right

            var xmax = x + 1;
            xmax = xmax != this.width ? xmax : 0; // wrap left

            var ymin = y - 1;
            ymin = ymin != -1 ? ymin : this.height - 1; // wrap top

            var ymax = y + 1;
            ymax = ymax != this.height ? ymax : 0; // wrap bottom

            var count = 0;

            count += this.cells[this.primary, xmin, ymin] == PixelState.Alive ? 1 : 0;
            count += this.cells[this.primary, xmin, y] == PixelState.Alive ? 1 : 0;
            count += this.cells[this.primary, xmin, ymax] == PixelState.Alive ? 1 : 0;

            count += this.cells[this.primary, xmax, ymin] == PixelState.Alive ? 1 : 0;
            count += this.cells[this.primary, xmax, y] == PixelState.Alive ? 1 : 0;
            count += this.cells[this.primary, xmax, ymax] == PixelState.Alive ? 1 : 0;

            count += this.cells[this.primary, x, ymin] == PixelState.Alive ? 1 : 0;
            count += this.cells[this.primary, x, ymax] == PixelState.Alive ? 1 : 0;

            return count;
        }

        public void Render(Graphics graphics)
        {
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            graphics.DrawImage(this.image, 0, 0, this.width * 8, this.height * 8);

            this.RunAsync();
        }

        private static Color GetColor(PixelState state)
        {
            return state switch
            {
                PixelState.Dying => Color.ForestGreen,
                PixelState.Alive => Color.Lime,
                _ => Color.Black,
            };
        }
    }
}

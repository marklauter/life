using System.Collections.Concurrent;

namespace GameOfLife
{
    internal class Simulation
    {
        // https://en.wikipedia.org/wiki/Conway's_Game_of_Life

        public long Generations { get; private set; } = 0;
        public event EventHandler<EventArgs>? FrameReady;

        private readonly ConcurrentQueue<Bitmap> bitmaps = new();
        private readonly Bitmap image;
        private readonly byte[,,] cells;
        private int primary = 0;
        private readonly int width;
        private readonly int height;

        public Simulation(int width, int height, int chance)
        {
            this.cells = new byte[2, width, height];
            this.image = new Bitmap(width, height);
            this.width = width;
            this.height = height;
            this.LetThereBeLight(chance);
        }

        public async void RunAsync(CancellationToken cancellationToken)
        {
            await Task.Run(async () =>
            {
                while (true)
                {
                    await Task.Delay(25);
                    this.ApplyRules();
                    this.RenderBitmap();
                    this.bitmaps.Enqueue((Bitmap)this.image.Clone());
                    ++this.Generations;
                    this.FrameReady?.Invoke(this, EventArgs.Empty);
                }
            }, cancellationToken);
        }

        public bool TryGetNextFrame(out Bitmap? frame)
        {
            return this.bitmaps.TryDequeue(out frame);
        }

        private void LetThereBeLight(int chance)
        {
            var rng = new Random(DateTime.UtcNow.Millisecond);
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    this.cells[this.primary, x, y] = rng.Next(20) <= 20 * (chance / 100.0)
                        ? (byte)0xFF
                        : (byte)0x00;
                }
            }
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

        private static Color GetColor(byte age)
        {
            return Color.FromArgb(age > 0 ? 255 - age : 0, age, 0);
        }

        private void ApplyRules()
        {
            var successor = this.primary ^ 1;
            var rng = new Random(DateTime.UtcNow.Millisecond);
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    var neighbors = this.CountLivingNeighbors(x, y);

                    // the RNG creates a chance that a cell will not die. this causes oscillators to sometimes spawn into puffers and gliders
                    this.cells[successor, x, y] = this.cells[this.primary, x, y] switch
                    {
                        0 => neighbors == 3 || (neighbors >= 2 && rng.NextDouble() < 0.0005) ? (byte)0xFF : (byte)0x00,
                        _ => neighbors == 2 || neighbors == 3 || (neighbors >= 2 && rng.NextDouble() < 0.0005) ? (byte)(this.cells[this.primary, x, y] - 1) : (byte)0x00,
                    };

                    if (this.cells[successor, x, y] < 0)
                    {
                        this.cells[successor, x, y] = 0x00;
                    }
                }
            }

            this.primary = successor;
        }

        private int CountLivingNeighbors(int x, int y)
        {
            var xmin = x > 0 ? x - 1 : this.width - 1; // wrap right
            var xmax = x < this.width - 1 ? x + 1 : 0; // wrap left
            var ymin = y > 0 ? y - 1 : this.height - 1;  // wrap top
            var ymax = y < this.height - 1 ? y + 1 : 0; // wrap bottom

            var count = 0;

            // value > zero is alive

            // left column
            count += this.cells[this.primary, xmin, ymin] > 0 ? 1 : 0;
            count += this.cells[this.primary, xmin, y] > 0 ? 1 : 0;
            count += this.cells[this.primary, xmin, ymax] > 0 ? 1 : 0;

            // right colum
            count += this.cells[this.primary, xmax, ymin] > 0 ? 1 : 0;
            count += this.cells[this.primary, xmax, y] > 0 ? 1 : 0;
            count += this.cells[this.primary, xmax, ymax] > 0 ? 1 : 0;

            // center column (excluding current pixel)
            count += this.cells[this.primary, x, ymin] > 0 ? 1 : 0;
            count += this.cells[this.primary, x, ymax] > 0 ? 1 : 0;

            return count;
        }
    }
}

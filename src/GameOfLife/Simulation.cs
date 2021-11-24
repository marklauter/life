using System.Runtime.InteropServices;

namespace GameOfLife
{
    internal class Simulation
    {
        // https://en.wikipedia.org/wiki/Conway's_Game_of_Life

        public long Generations { get; private set; } = 0;
        public event EventHandler<EventArgs>? FrameReady;

        private readonly Bitmap frame;
        private readonly Rectangle frameBounds;
        private readonly int depth;
        private readonly byte[][] cells;
        private int primary = 0;
        private readonly int width;
        private readonly int height;

        public Simulation(int width, int height, int chance)
        {
            this.cells = new byte[2][];
            this.cells[0] = new byte[width * height];
            this.cells[1] = new byte[width * height];
            this.frame = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            this.frameBounds = new Rectangle(0, 0, width, height);
            this.depth = Bitmap.GetPixelFormatSize(this.frame.PixelFormat);
            this.width = width;
            this.height = height;
            this.LetThereBeLight(chance);
        }

        public void GenerateFrameAsync()
        {
            this.WriteFrame();
            ++this.Generations;
            this.FrameReady?.Invoke(this, EventArgs.Empty);
        }

        public void DrawFrame(Graphics graphics)
        {
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            graphics.DrawImage(this.frame, 0, 0, this.width * 2, this.height * 2);
            this.GenerateFrameAsync();
        }

        private void LetThereBeLight(int chance)
        {
            var rng = new Random(DateTime.UtcNow.Millisecond);
            var bitmap = this.frame.LockBits(this.frameBounds, System.Drawing.Imaging.ImageLockMode.WriteOnly, this.frame.PixelFormat);
            try
            {
                var ptr = bitmap.Scan0;
                Marshal.Copy(ptr, this.cells[this.primary], 0, this.cells[this.primary].Length);

                for (var x = 0; x < this.width; ++x)
                {
                    for (var y = 0; y < this.height; ++y)
                    {
                        this.cells[this.primary][y * this.width + x] = rng.Next(20) <= 20 * (chance / 100.0)
                            ? (byte)0xFF
                            : (byte)0x00;
                    }
                }

                Marshal.Copy(this.cells[this.primary], 0, bitmap.Scan0, this.cells[this.primary].Length);
            }
            finally
            {
                this.frame.UnlockBits(bitmap);
            }
        }

        private void WriteFrame()
        {
            var successor = this.primary ^ 1;
            var bitmap = this.frame.LockBits(this.frameBounds, System.Drawing.Imaging.ImageLockMode.WriteOnly, this.frame.PixelFormat);
            try
            {
                Marshal.Copy(bitmap.Scan0, this.cells[this.primary], 0, this.cells[this.primary].Length);
                this.ApplyRules();
                Marshal.Copy(this.cells[successor], 0, bitmap.Scan0, this.cells[successor].Length);
                this.primary = successor;
            }
            finally
            {
                this.frame.UnlockBits(bitmap);
            }
        }

        private void ApplyRules()
        {
            var successor = this.primary ^ 1;
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    var currentPixel = y * this.width + x;
                    var neighbors = this.CountLivingNeighbors(x, y);

                    this.cells[successor][currentPixel] = this.cells[this.primary][currentPixel] switch
                    {
                        0 => neighbors == 3
                            ? (byte)0xFF
                            : (byte)0x00,
                        _ => neighbors == 2 || neighbors == 3
                            ? (byte)0xFF
                            : (byte)0x00,
                    };
                }
            }
        }

        private int CountLivingNeighbors(int x, int y)
        {
            var xmin = x > 0 ? x - 1 : this.width - 1; // wrap right
            var xmax = x < this.width - 1 ? x + 1 : 0; // wrap left
            var ymin = y > 0 ? y - 1 : this.height - 1;  // wrap top
            var ymax = y < this.height - 1 ? y + 1 : 0; // wrap bottom

            var count = 0;

            // value != zero is alive

            // left column
            count += this.cells[this.primary][xmin + this.width * ymin] != 0 ? 1 : 0;
            count += this.cells[this.primary][xmin + this.width * y] != 0 ? 1 : 0;
            count += this.cells[this.primary][xmin + this.width * ymax] != 0 ? 1 : 0;

            // right colum
            count += this.cells[this.primary][xmax + this.width * ymin] != 0 ? 1 : 0;
            count += this.cells[this.primary][xmax + this.width * y] != 0 ? 1 : 0;
            count += this.cells[this.primary][xmax + this.width * ymax] != 0 ? 1 : 0;

            // center column (excluding current pixel)
            count += this.cells[this.primary][x + this.width * ymin] != 0 ? 1 : 0;
            count += this.cells[this.primary][x + this.width * ymax] != 0 ? 1 : 0;

            return count;
        }
    }
}

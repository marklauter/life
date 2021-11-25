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
        private readonly byte[][] cells;
        private int source = 0;
        private readonly int width;
        private readonly int height;
        private readonly Graphics graphics;

        public Simulation(int width, int height, int chance, Graphics graphics)
        {
            this.cells = new byte[2][];
            this.cells[0] = new byte[width * height];
            this.cells[1] = new byte[width * height];
            this.frame = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            this.frameBounds = new Rectangle(0, 0, width, height);
            this.width = width;
            this.height = height;
            this.graphics = graphics;
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
            graphics.DrawImage(this.frame, 0, 0, this.width, this.height);
            this.GenerateFrameAsync();
        }

        private void LetThereBeLight(int chance)
        {
            var rng = new Random(DateTime.UtcNow.Millisecond);
            var bitmap = this.frame.LockBits(this.frameBounds, System.Drawing.Imaging.ImageLockMode.WriteOnly, this.frame.PixelFormat);
            try
            {
                var value = (byte)0xFF;
                for (var x = 0; x < this.width; ++x)
                {
                    for (var y = 0; y < this.height; ++y)
                    {
                        value = (rng.NextDouble() * 100 <= chance + (value == 0xFF ? chance * 0.50 : 0))
                            ? (byte)0xFF
                            : (byte)0x00;
                        this.cells[this.source][y * this.width + x] = value;
                    }
                }

                Marshal.Copy(this.cells[this.source], 0, bitmap.Scan0, this.cells[this.source].Length);
            }
            finally
            {
                this.frame.UnlockBits(bitmap);
            }
        }

        private void WriteFrame()
        {
            var target = this.source ^ 1;
            var bitmap = this.frame.LockBits(this.frameBounds, System.Drawing.Imaging.ImageLockMode.WriteOnly, this.frame.PixelFormat);
            try
            {
                this.ApplyRules(target);
                Marshal.Copy(this.cells[target], 0, bitmap.Scan0, this.cells[target].Length);
            }
            finally
            {
                this.frame.UnlockBits(bitmap);
            }

            this.source = target;
        }

        private void ApplyRules(int target)
        {
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    var currentPixel = y * this.width + x;
                    var neighbors = this.CountLivingNeighbors(x, y);
                    var alive = this.cells[this.source][currentPixel] != 0x00;
                    var cell = this.cells[this.source][currentPixel];
                    // SR 2
                    // 0000 0000 1 
                    // 0000 0000 2 
                    // 0000 0000 3
                    // 0000 0001 4
                    // 0000 0001 5
                    // 0000 0001 6
                    // 0000 0001 7
                    // 0000 0010 8
                    // XOR 0x01
                    // 0000 0001 1 
                    // 0000 0001 2 
                    // 0000 0001 3
                    // 0000 0000 4
                    // 0000 0000 5
                    // 0000 0000 6
                    // 0000 0000 7
                    // 0000 0011 8
                    // SL 1
                    // 0000 0010 1 
                    // 0000 0010 2 
                    // 0000 0010 3
                    // 0000 0000 4
                    // 0000 0000 5
                    // 0000 0000 6
                    // 0000 0000 7
                    // 0000 0110 8
                    // AND self
                    // 0000 0000 1 
                    // 0000 0010 2 
                    // 0000 0010 3
                    // 0000 0000 4
                    // 0000 0000 5
                    // 0000 0000 6
                    // 0000 0000 7
                    // 0000 0000 8
                    // SR 1
                    // 0000 0000 1 
                    // 0000 0001 2 
                    // 0000 0001 3
                    // 0000 0000 4
                    // 0000 0000 5
                    // 0000 0000 6
                    // 0000 0000 7
                    // 0000 0000 8

                    // when neighbors is 2 then final multiplication zeros the value unless cell was alive, when it's 3 then it keeps the value
                    var nextValue = (byte)(
                        (((((neighbors >> 2) ^ 0x01) << 1) & neighbors) >> 1)
                        * ((neighbors & 0x01) | (cell & 0x01))
                        * 0xFF); // answer is either FF or 00


                    //var nextValue = (alive && neighbors == 2) || neighbors == 3
                    //    ? (byte)0xFF
                    //    : (byte)0x00;

                    //if (shiftedValue != nextValue)
                    //{
                    //    throw new InvalidOperationException();
                    //}

                    this.cells[target][currentPixel] = nextValue;
                }
            }
        }

        private int CountLivingNeighbors(int x, int y)
        {
            // wrap on the edges
            var xmin = x > 0 ? x - 1 : this.width - 1;
            var xmax = x < this.width - 1 ? x + 1 : 0;
            var ymin = y > 0 ? y - 1 : this.height - 1;
            var ymax = y < this.height - 1 ? y + 1 : 0;

            var count = 0;

            // cells can be 0xFF or 0x00. Any non-zero value is alive

            // left column
            count += this.cells[this.source][xmin + this.width * ymin];
            count += this.cells[this.source][xmin + this.width * y];
            count += this.cells[this.source][xmin + this.width * ymax];

            // right colum
            count += this.cells[this.source][xmax + this.width * ymin];
            count += this.cells[this.source][xmax + this.width * y];
            count += this.cells[this.source][xmax + this.width * ymax];

            // center column (excluding current pixel)
            count += this.cells[this.source][x + this.width * ymin];
            count += this.cells[this.source][x + this.width * ymax];

            count /= 0xFF;

            return count;
        }
    }
}

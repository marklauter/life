using System.Runtime.CompilerServices;
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
        private readonly int[][] neighbors;
        private int source = 0;
        private readonly int width;
        private readonly int height;
        private readonly int magnifier;

        public Simulation(int width, int height, int magnifier)
        {
            this.width = width;
            this.height = height;
            this.magnifier = magnifier;
            this.cells = new byte[2][];
            this.cells[0] = new byte[width * height];
            this.cells[1] = new byte[width * height];
            this.neighbors = new int[width * height][];
            this.PrecomputeNeighbors();
            this.frame = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            this.frameBounds = new Rectangle(0, 0, width, height);
        }

        private void PrecomputeNeighbors()
        {
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    var xmin = x > 0 ? x - 1 : this.width - 1;
                    var xmax = x < this.width - 1 ? x + 1 : 0;
                    var ymin = y > 0 ? y - 1 : this.height - 1;
                    var ymax = y < this.height - 1 ? y + 1 : 0;
                    var currentPixel = y * this.width + x;

                    this.neighbors[currentPixel] = new int[8];
                    this.neighbors[currentPixel][0] = xmin + this.width * ymin;
                    this.neighbors[currentPixel][1] = xmin + this.width * y;
                    this.neighbors[currentPixel][2] = xmin + this.width * ymax;
                    this.neighbors[currentPixel][3] = xmax + this.width * ymin;
                    this.neighbors[currentPixel][4] = xmax + this.width * y;
                    this.neighbors[currentPixel][5] = xmax + this.width * ymax;
                    this.neighbors[currentPixel][6] = x + this.width * ymin;
                    this.neighbors[currentPixel][7] = x + this.width * ymax;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void GenerateFrameAsync()
        {
            var target = this.source ^ 1;
            this.ApplyRules(target);
            this.WriteFrame(target);
            this.source = target;
            ++this.Generations;
            this.FrameReady?.Invoke(this, EventArgs.Empty);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void DrawFrame(Graphics graphics)
        {
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            graphics.DrawImage(this.frame, 0, 0, this.width * this.magnifier, this.height * this.magnifier);
            this.GenerateFrameAsync();
        }

        public void LetThereBeLight((int x, int y)[] initialState)
        {
            var stateWidth = initialState.Max(t => t.x) - initialState.Min(t => t.x);
            var stateHeight = initialState.Max(t => t.y) - initialState.Min(t => t.y);
            var offset = (x: this.width / 2 - stateWidth / 2, y: this.height / 2 - stateHeight / 2);

            for (var i = 0; i < initialState.Length; ++i)
            {
                var state = initialState[i];
                var x = state.x + offset.x;
                var y = state.y + offset.y;
                this.cells[this.source][y * this.width + x] = 0xFF;
            }
        }

        public void LetThereBeLight(Image image)
        {
            var depth = Bitmap.GetPixelFormatSize(System.Drawing.Imaging.PixelFormat.Format24bppRgb) / 8;

            using var resizedImage = new Bitmap(this.width, this.height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            using var graphics = Graphics.FromImage(resizedImage);
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            graphics.DrawImage(image, 0, 0, this.width, this.height);

            var bitmap = resizedImage.LockBits(this.frameBounds, System.Drawing.Imaging.ImageLockMode.ReadOnly, resizedImage.PixelFormat);
            var rgb = new byte[bitmap.Stride * this.height];
            Marshal.Copy(bitmap.Scan0, rgb, 0, rgb.Length);

            var targetCells = this.cells[this.source];

            var cell = 0;
            for (var i = 0; i < rgb.Length; i += depth)
            {
                var avg = (rgb[i] + rgb[i + 1] + rgb[i + 2]) / 3;
                var value = (byte)(avg >= 175 ? 0xFF : 0x00);
                targetCells[cell++] = value;
            }

            resizedImage.UnlockBits(bitmap);
        }

        public void LetThereBeLight(int chance)
        {
            var rng = new Random(DateTime.UtcNow.Millisecond);
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    this.cells[this.source][y * this.width + x] = rng.NextDouble() * 100 <= chance
                        ? (byte)0xFF
                        : (byte)0x00;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void ApplyRules(int target)
        {
            var sourceCells = this.cells[this.source];
            var targetCells = this.cells[target];

            //for (var currentPixel = 0; currentPixel < sourceCells.Length; ++currentPixel)
            //{
            //    var neighbors = this.CountLivingNeighbors(currentPixel, sourceCells);
            //    targetCells[currentPixel] = (byte)(
            //        (((((neighbors >> 2) ^ 0x01) << 1) & neighbors) >> 1)
            //        * ((neighbors & 0x01) | (sourceCells[currentPixel] & 0x01))
            //        * 0xFF);
            //}

            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    var neighbors = this.CountLivingNeighbors(x, y, sourceCells);

                    var currentPixel = y * this.width + x;

                    // this is all the rules of life in 1 line of code - no conditionals
                    targetCells[currentPixel] = (byte)(
                        (((((neighbors >> 2) ^ 0x01) << 1) & neighbors) >> 1)
                        * ((neighbors & 0x01) | (sourceCells[currentPixel] & 0x01))
                        * 0xFF);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private int CountLivingNeighbors(int currentPixel, byte[] sourceCells)
        {
            var neighbors = this.neighbors[currentPixel];

            return
                // left column
                (sourceCells[neighbors[0]]
                + sourceCells[neighbors[1]]
                + sourceCells[neighbors[2]]
                // right colum
                + sourceCells[neighbors[3]]
                + sourceCells[neighbors[4]]
                + sourceCells[neighbors[5]]
                // center column (excluding current pixel)
                + sourceCells[neighbors[6]]
                + sourceCells[neighbors[7]])
                / 0xFF;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private int CountLivingNeighbors(int x, int y, byte[] sourceCells)
        {
            // note: I tried precalculating the neighbors and adding them to a lookup array, but this was slower than calculating on the fly. that was a surprise.

            // wrap on the edges
            var xmin = x > 0 ? x - 1 : this.width - 1;
            var xmax = x < this.width - 1 ? x + 1 : 0;
            var ymin = y > 0 ? y - 1 : this.height - 1;
            var ymax = y < this.height - 1 ? y + 1 : 0;

            // cells can be 0xFF or 0x00

            return
                // left column
                (sourceCells[xmin + this.width * ymin]
                + sourceCells[xmin + this.width * y]
                + sourceCells[xmin + this.width * ymax]
                // right colum
                + sourceCells[xmax + this.width * ymin]
                + sourceCells[xmax + this.width * y]
                + sourceCells[xmax + this.width * ymax]
                // center column (excluding current pixel)
                + sourceCells[x + this.width * ymin]
                + sourceCells[x + this.width * ymax])
                / 0xFF;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void WriteFrame(int target)
        {
            var targetCells = this.cells[target];
            var bitmap = this.frame.LockBits(this.frameBounds, System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            Marshal.Copy(targetCells, 0, bitmap.Scan0, targetCells.Length);
            this.frame.UnlockBits(bitmap);
        }
    }
}


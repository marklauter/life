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
        private int source = 0;
        private readonly int width;
        private readonly int height;

        public Simulation(int width, int height)
        {
            this.cells = new byte[2][];
            this.cells[0] = new byte[width * height];
            this.cells[1] = new byte[width * height];
            this.frame = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            this.frameBounds = new Rectangle(0, 0, width, height);

            this.width = width;
            this.height = height;
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
            graphics.DrawImage(this.frame, 0, 0, this.width * 2, this.height * 2);
            this.GenerateFrameAsync();
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
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    var neighbors = this.CountLivingNeighbors(x, y);
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

                    var currentPixel = y * this.width + x;

                    // this is all the rules of life in 1 line of code
                    targetCells[currentPixel] = (byte)(
                        (((((neighbors >> 2) ^ 0x01) << 1) & neighbors) >> 1)
                        * ((neighbors & 0x01) | (sourceCells[currentPixel] & 0x01))
                        * 0xFF);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private int CountLivingNeighbors(int x, int y)
        {
            // note: I tried precalculating the neighbors and adding them to a lookup array, but this was slower than calculating on the fly. that was a surprise.

            // wrap on the edges
            var xmin = x > 0 ? x - 1 : this.width - 1;
            var xmax = x < this.width - 1 ? x + 1 : 0;
            var ymin = y > 0 ? y - 1 : this.height - 1;
            var ymax = y < this.height - 1 ? y + 1 : 0;

            var sourceCells = this.cells[this.source];
            var count = 0;

            // cells can be 0xFF or 0x00

            // left column
            count += sourceCells[xmin + this.width * ymin];
            count += sourceCells[xmin + this.width * y];
            count += sourceCells[xmin + this.width * ymax];

            // right colum
            count += sourceCells[xmax + this.width * ymin];
            count += sourceCells[xmax + this.width * y];
            count += sourceCells[xmax + this.width * ymax];

            // center column (excluding current pixel)
            count += sourceCells[x + this.width * ymin];
            count += sourceCells[x + this.width * ymax];

            count /= 0xFF;

            return count;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void WriteFrame(int target)
        {
            var targetCells = this.cells[target];
            var bitmap = this.frame.LockBits(this.frameBounds, System.Drawing.Imaging.ImageLockMode.WriteOnly, this.frame.PixelFormat);
            Marshal.Copy(targetCells, 0, bitmap.Scan0, targetCells.Length);
            this.frame.UnlockBits(bitmap);
        }
    }
}

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace GameOfLife
{
    internal class SimulationV2
    {
        // https://en.wikipedia.org/wiki/Conway's_Game_of_Life

        public long Generations { get; private set; } = 0;
        public event EventHandler<EventArgs>? FrameReady;

        private readonly Bitmap frame;
        private readonly Rectangle frameBounds;
        private readonly byte[] cells;
        private readonly int[][] neighborOffsets;
        private HashSet<int> activeCells = new();
        private readonly int width;
        private readonly int height;
        private readonly int span;
        private readonly int magnifier;

        public SimulationV2(int width, int height, int magnifier)
        {
            this.width = width;
            this.height = height;
            this.span = width * height;
            this.magnifier = magnifier;

            this.cells = new byte[this.span];

            this.neighborOffsets = new int[this.span][];
            this.PrecomputeNeighborOffsets();
            this.frame = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            this.frameBounds = new Rectangle(0, 0, width, height);
        }

        private void PrecomputeNeighborOffsets()
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

                    this.neighborOffsets[currentPixel] = new int[8];
                    this.neighborOffsets[currentPixel][0] = xmin + this.width * ymin;
                    this.neighborOffsets[currentPixel][1] = xmin + this.width * y;
                    this.neighborOffsets[currentPixel][2] = xmin + this.width * ymax;
                    this.neighborOffsets[currentPixel][3] = xmax + this.width * ymin;
                    this.neighborOffsets[currentPixel][4] = xmax + this.width * y;
                    this.neighborOffsets[currentPixel][5] = xmax + this.width * ymax;
                    this.neighborOffsets[currentPixel][6] = x + this.width * ymin;
                    this.neighborOffsets[currentPixel][7] = x + this.width * ymax;
                }
            }
        }

        private void InitializeActiveCells()
        {
            for (var i = 0; i < this.span; ++i)
            {
                if (this.cells[i] == 0xFF)
                {
                    this.activeCells.Add(i);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void GenerateFrame()
        {
            this.ApplyRules();
            this.WriteFrame();
            ++this.Generations;
            this.FrameReady?.Invoke(this, EventArgs.Empty);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void DrawFrame(Graphics graphics)
        {
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            graphics.DrawImage(this.frame, 0, 0, this.width * this.magnifier, this.height * this.magnifier);
            this.GenerateFrame();
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
                this.cells[y * this.width + x] = 0xFF;
            }

            this.InitializeActiveCells();
        }

        public void LetThereBeLight(int chance)
        {
            var rng = new Random(DateTime.UtcNow.Millisecond);
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    this.cells[y * this.width + x] = rng.NextDouble() * 100 <= chance
                        ? (byte)0xFF
                        : (byte)0x00;
                }
            }

            this.InitializeActiveCells();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void ApplyRules()
        {
            var newActiveCells = new HashSet<int>();
            var changedCells = new List<(int offset, byte value)>(this.activeCells.Count);
            var visited = new HashSet<int>();

            foreach (var cellOffset in this.activeCells)
            {
                var offset = cellOffset;
                var outerLocalNeighborOffsets = this.neighborOffsets[offset];

                if (!visited.Contains(offset))
                {
                    visited.Add(offset);

                    var originalValue = this.cells[offset];
                    var liveNeighborCount =
                        // left column
                        (this.cells[outerLocalNeighborOffsets[0]]
                        + this.cells[outerLocalNeighborOffsets[1]]
                        + this.cells[outerLocalNeighborOffsets[2]]
                        // right colum
                        + this.cells[outerLocalNeighborOffsets[3]]
                        + this.cells[outerLocalNeighborOffsets[4]]
                        + this.cells[outerLocalNeighborOffsets[5]]
                        // center column (excluding current pixel)
                        + this.cells[outerLocalNeighborOffsets[6]]
                        + this.cells[outerLocalNeighborOffsets[7]])
                        / 0xFF;

                    var newValue = (byte)(
                        (((((liveNeighborCount >> 2) ^ 0x01) << 1) & liveNeighborCount) >> 1)
                        * ((liveNeighborCount & 0x01) | (originalValue & 0x01))
                        * 0xFF);

                    if (originalValue != newValue)
                    {
                        changedCells.Add((offset, newValue));
                    }

                    if (newValue == 0xFF)
                    {
                        newActiveCells.Add(offset);
                    }
                }

                for (var j = 0; j < 8; ++j)
                {
                    offset = outerLocalNeighborOffsets[j];

                    if (!visited.Contains(offset))
                    {
                        visited.Add(offset);

                        var originalValue = this.cells[offset];
                        var innerLocalNeighborOffsets = this.neighborOffsets[offset];
                        var liveNeighborCount =
                            // left column
                            (this.cells[innerLocalNeighborOffsets[0]]
                            + this.cells[innerLocalNeighborOffsets[1]]
                            + this.cells[innerLocalNeighborOffsets[2]]
                            // right colum
                            + this.cells[innerLocalNeighborOffsets[3]]
                            + this.cells[innerLocalNeighborOffsets[4]]
                            + this.cells[innerLocalNeighborOffsets[5]]
                            // center column (excluding current pixel)
                            + this.cells[innerLocalNeighborOffsets[6]]
                            + this.cells[innerLocalNeighborOffsets[7]])
                            / 0xFF;

                        var newValue = (byte)(
                            (((((liveNeighborCount >> 2) ^ 0x01) << 1) & liveNeighborCount) >> 1)
                            * ((liveNeighborCount & 0x01) | (originalValue & 0x01))
                            * 0xFF);

                        if (originalValue != newValue)
                        {
                            changedCells.Add((offset, newValue));
                        }

                        if (newValue == 0xFF)
                        {
                            newActiveCells.Add(offset);
                        }
                    }
                }
            }

            foreach (var (offset, value) in changedCells)
            {
                this.cells[offset] = value;
            }

            this.activeCells = newActiveCells;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void WriteFrame()
        {
            var bitmap = this.frame.LockBits(this.frameBounds, System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            Marshal.Copy(this.cells, 0, bitmap.Scan0, this.cells.Length);
            this.frame.UnlockBits(bitmap);
        }
    }
}


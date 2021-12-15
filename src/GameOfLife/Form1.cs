using Newtonsoft.Json;
using System.Diagnostics;

namespace GameOfLife
{
    public partial class Life : Form
    {
        private readonly Simulation simulation;
        private readonly Stopwatch stopwatch = new();
        private long framecount = 0;

        public Life()
        {
            this.InitializeComponent();

            //this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
            this.Width = 1200;
            this.Height = 800;

            this.simulation = new(this.Width / 4, this.Height / 4);
            this.simulation.FrameReady += this.Simulation_FrameReady;
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);

            var json = File.ReadAllText("initialState.json");
            var initialState = JsonConvert.DeserializeObject<(int x, int y)[]>(json);
            if (initialState == null)
            {
                throw new FileFormatException();
            }

            this.simulation.LetThereBeLight(initialState);

            //this.simulation.LetThereBeLight(7);
            //this.simulation.LetThereBeLight(Bitmap.FromFile("cortana.jpg"));

            this.stopwatch.Start();
            this.simulation.GenerateFrameAsync();

            //var initialState = new (int x, int y)[36];
            //initialState[0] = (1, 5);
            //initialState[1] = (1, 6);
            //initialState[2] = (2, 5);
            //initialState[3] = (2, 6);
            //var json = JsonConvert.SerializeObject(initialState);
        }

        private void Simulation_FrameReady(object? sender, EventArgs e)
        {
            ++this.framecount;
            var fps = (int)(this.framecount / this.stopwatch.Elapsed.TotalSeconds);
            this.Text = $"Life - Generations ({this.simulation.Generations}), FPS ({fps})";
            this.Invalidate();
        }

        private void Life_Paint(object sender, PaintEventArgs e)
        {
            this.simulation.DrawFrame(e.Graphics);
        }
    }
}
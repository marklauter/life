using Newtonsoft.Json;
using System.Diagnostics;

namespace GameOfLife
{
    public partial class Life : Form
    {
        private readonly SimulationV2 simulation;
        private readonly Stopwatch stopwatch = new();
        private readonly int magnifier = 3;

        public Life()
        {
            this.InitializeComponent();

            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
            this.Width = 1200;
            this.Height = 900;

            this.simulation = new(
                this.Width / this.magnifier,
                this.Height / this.magnifier,
                this.magnifier);
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

            this.stopwatch.Start();
            this.simulation.GenerateFrame();
        }

        private void Simulation_FrameReady(object? sender, EventArgs e)
        {
            if (this.simulation.Generations % 100 == 0)
            {
                var fps = (int)(this.simulation.Generations / this.stopwatch.Elapsed.TotalSeconds);
                this.Text = $"Life - Generations ({this.simulation.Generations}), FPS ({fps})";
            }

            this.Invalidate();
        }

        private void Life_Paint(object sender, PaintEventArgs e)
        {
            this.simulation.DrawFrame(e.Graphics);
        }

        private void Life_Click(object sender, EventArgs e)
        {
            this.Close();
        }
    }
}
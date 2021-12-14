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
            this.Width = 1600;
            this.Height = 800;

            this.simulation = new(this.Width / 2, this.Height / 2);
            this.simulation.FrameReady += this.Simulation_FrameReady;
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);

            this.simulation.LetThereBeLight(7);
            //this.simulation.LetThereBeLight(Bitmap.FromFile("cortana.jpg"));
            this.stopwatch.Start();
            this.simulation.GenerateFrameAsync();
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
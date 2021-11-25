using System.Runtime.InteropServices;

namespace GameOfLife
{
    public partial class Life : Form
    {
        private readonly Simulation simulation;

        public Life()
        {
            this.InitializeComponent();
            this.simulation = new(this.canvas.Width / 4, this.canvas.Height / 4);
            this.simulation.FrameReady += this.Simulation_FrameReady;
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            this.simulation.LetThereBeLight(10);
            //this.simulation.LetThereBeLight(Bitmap.FromFile("cortana.jpg"));
            this.simulation.GenerateFrameAsync();
        }

        private void Simulation_FrameReady(object? sender, EventArgs e)
        {
            this.Text = $"Life - Generations ({this.simulation.Generations})";
            this.canvas.Invalidate();
        }

        private void Canvas_Paint(object sender, PaintEventArgs e)
        {
            this.simulation.DrawFrame(e.Graphics);
        }
    }
}
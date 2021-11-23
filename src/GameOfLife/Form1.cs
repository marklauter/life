using System.ComponentModel;

namespace GameOfLife
{
    public partial class Life : Form
    {
        private readonly Simulation simulation;
        private readonly CancellationTokenSource simulationCancellationSource;

        public Life()
        {
            this.InitializeComponent();
            this.simulationCancellationSource = new CancellationTokenSource();
            this.simulation = new(this.canvas.Width / 8, this.canvas.Height / 8, 5);
            this.simulation.FrameReady += this.Simulation_FrameReady;
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            this.simulation.RunAsync(this.simulationCancellationSource.Token);
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            this.simulationCancellationSource.Cancel();
            base.OnClosing(e);
        }

        private void Simulation_FrameReady(object? sender, EventArgs e)
        {
            this.BeginInvoke(() =>
            {
                this.Text = $"Life - Generations ({this.simulation.Generations})";
                this.canvas.Invalidate();
            });
        }

        private void Canvas_Paint(object sender, PaintEventArgs e)
        {
            this.Render(e.Graphics);
        }

        private void Render(Graphics graphics)
        {
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            if (this.simulation.TryGetNextFrame(out var frame) && frame != null)
            {
                graphics.DrawImage(frame, 0, 0, this.canvas.Width, this.canvas.Height);
                frame.Dispose();
            }
        }
    }
}
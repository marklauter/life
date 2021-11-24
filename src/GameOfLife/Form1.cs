namespace GameOfLife
{
    public partial class Life : Form
    {
        private readonly Simulation simulation;

        public Life()
        {
            this.InitializeComponent();
            this.simulation = new(this.canvas.Width / 2, this.canvas.Height / 2, 5);
            this.simulation.FrameReady += this.Simulation_FrameReady;
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
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
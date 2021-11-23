namespace GameOfLife
{
    public partial class Life : Form
    {
        private Simulation simulation;

        public Life()
        {
            this.InitializeComponent();
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            this.simulation = new(this.canvas.Width / 8, this.canvas.Height / 8, 5);
            this.simulation.SimulationChanged += this.Simulation_SimulationChanged;
        }

        private void Simulation_SimulationChanged(object? sender, EventArgs e)
        {
            this.canvas.Invalidate();
            this.Text = $"Life - Generations ({this.simulation.Generations})";
        }

        private void Canvas_Paint(object sender, PaintEventArgs e)
        {
            this.simulation.Render(e.Graphics);
        }
    }
}
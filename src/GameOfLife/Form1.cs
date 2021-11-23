namespace GameOfLife
{
    public partial class Form1 : Form
    {
        private Simulation simulation;

        public Form1()
        {
            this.InitializeComponent();
            this.DoubleBuffered = true;
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            this.simulation = new(this.canvas.Width / 4, this.canvas.Height / 4, 5);
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
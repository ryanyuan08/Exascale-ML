# 1D Burgers' Equation Solver

A C++ implementation of a 1D Burgers' equation solver using finite difference methods with support for multiple numerical schemes, artificial viscosity, and automated training data generation. This solver can simulate both inviscid and viscous Burgers' equations with customizable or randomly generated initial conditions.

## What is Burgers' Equation?

Burgers' equation is a fundamental partial differential equation used in fluid mechanics that models how velocity fields evolve over time. It combines:
- **Convection**: How the wave shape moves and steepens
- **Diffusion**: How viscosity smooths out sharp features

This makes it useful for understanding shock waves, traffic flow, and basic fluid dynamics.

## Project Structure

```
.
├── include/
│   ├── burgers.h                      # BurgersSolver1d class definition
│   ├── burger_scheme.h               # Abstract scheme interface and implementations
│   └── initial_condition_generator.h  # Random initial condition generator
├── src/
│   ├── main.cpp                       # Example simulations
│   ├── burgers.cpp                    # Solver implementation
│   ├── training_data_generator.cpp    # Automated training data generation
│   ├── initial_condition_generator.cpp # Random IC implementation
│   └── schemes/
│       ├── ftcs.cpp                   # Forward-Time Central-Space scheme
│       └── lax_wendroff.cpp          # Lax-Wendroff scheme
├── visualizer/
│   └── visualize.py                   # Interactive visualization tool
├── CMakeLists.txt                     # Build configuration
├── requirements.txt                   # Python dependencies
├── data/                              # Output directory for experiments
└── training_data/                     # Output directory for training samples
```

## Building the Project

### Prerequisites
- CMake 3.10 or higher
- C++20 compatible compiler (GCC, Clang, or MSVC)
- Python 3.x (for visualization)

### Build Instructions

1. Create a build directory and navigate to it:
```bash
mkdir build
cd build
```

2. Generate the build files with CMake:
```bash
cmake ..
```

3. Compile the project:
```bash
make
```

4. Run the executables:
```bash
# Run example simulations
./burgers

# Generate training data
./training_data_generator
```

The programs will generate output in `../data` and `../training_data` directories.

## Numerical Schemes

The solver supports multiple finite difference schemes through a polymorphic scheme architecture:

### Available Schemes

#### 1. FTCS (Forward-Time Central-Space)
- Simple explicit scheme
- First-order accurate in time, second-order in space
- Includes artificial viscosity for shock capturing
- Good for smooth solutions

#### 2. Lax-Wendroff
- Two-step predictor-corrector method
- Second-order accurate in both time and space
- Better for capturing sharp gradients and shocks
- More computationally expensive than FTCS

Both schemes implement artificial viscosity to stabilize solutions near shocks:
```cpp
// Applied only in compression regions (∂u/∂x < 0)
artificial_viscosity = cq * dx² * |∂u/∂x|
```
#### 3. Godunov
- Riemann solving for flux model
- Accurate; doesn't need artificial viscosity
- WAY more computationally expensive than FTCS, especially when you get to more complex equations.

## How to Use

### Creating a Simulation

#### 1. Solver Configuration

Define a `SolverConfig` struct:

```cpp
SolverConfig config = {
    .kinematic_viscosity = 0.01,    // Physical viscosity coefficient (ν)
    .time_steps = 2000,             // Number of time iterations
    .time_step_size = 0.001,        // Time step size (Δt)
    .domain_length = 2.0            // Length of spatial domain [0, L]
};
```

**Parameter Explanation:**
- `kinematic_viscosity`: Controls physical diffusion (higher = more smoothing)
- `time_steps`: Number of temporal iterations
- `time_step_size`: Time increment between updates
- `domain_length`: Spatial domain interval [0, L]

**Note**: The spatial grid resolution is automatically computed based on stability requirements using `ALPHA = 0.9` and the maximum velocity in the initial condition.

#### 2. Initial Condition Function

Define a function specifying the initial velocity profile:

```cpp
// Step function
std::function<double(double)> step_function = [](double x) -> double {
    if (x >= 0.5 && x <= 1.0) {
        return 2.0;
    } else {
        return 1.0;
    }
};

// Sine wave
std::function<double(double)> sine_function = [](double x) -> double {
    return std::sin(x);
};
```

#### 3. Create and Run the Solver

```cpp
// Create solver with chosen scheme
BurgersSolver1d solver(
    std::make_unique<LaxWendroff>(),  // or std::make_unique<FTCS>()
    config,
    step_function
);

// Run simulation with artificial viscosity coefficient
solver.solve(2.0);  // cq = 2.0

// Save results (base_folder, run_name, gap)
solver.saveSolution("../data", "step_function", 10);
```

The `gap` parameter controls output frequency (e.g., `10` saves every 10th timestep).

### Random Initial Conditions

Generate random superpositions of sinusoids for training data:

```cpp
// Configure random initial condition generator
RandomInitialConditionConfig config;
config.n = 5;                               // Number of sinusoidal terms
config.domain_length = 10.0;                // Domain size
config.amp_min = -1.0;                      // Min amplitude
config.amp_max = 1.0;                       // Max amplitude
config.frequency_multiplier_min = 0.1;      // Min frequency multiplier
config.frequency_multiplier_max = 5.0;      // Max frequency multiplier
config.wrap_around_frequency_multiplier_min = 1;   // Min Fourier mode
config.wrap_around_frequency_multiplier_max = 5;   // Max Fourier mode

// Create random IC (config, alwaysPositive, wrapAround, seed)
RandomInitialCondition f(config, false, true);

// Use in solver
BurgersSolver1d solver(
    std::make_unique<FTCS>(),
    solver_config,
    f
);

solver.solve();
solver.saveSolution("../data", "random_function", 1);
```

**Random IC Parameters:**
- `alwaysPositive`: If true, adds bias to ensure f(x) ≥ 0
- `wrapAround`: If true, uses Fourier frequencies (2πk/L) for periodic BCs
- `seed`: Random seed for reproducibility

The generated function has the form:
```
f(x) = bias + Σ[i=1 to n] A_i * sin(ω_i * (x - φ_i))
```

## Training Data Generation

The `training_data_generator` executable creates datasets for machine learning applications. It supports both fresh generation and appending to existing datasets through command line arguments.

### Default Configuration

The following constants in `training_data_generator.cpp` serve as defaults when not overridden by command line arguments:

```cpp
const int DEFAULT_NUM_SAMPLES = 2;      // Number of samples to generate
const std::string TRAINING_DIR = "../training_data";

// Solver parameters
const double KINEMATIC_VISCOSITY = 0.01;
const int DEFAULT_TIME_STEPS = 100000;
const double DEFAULT_TIME_STEP_SIZE = 0.000001;

// Random IC ranges
const int N_MIN = 1;                    // Min number of terms
const int N_MAX = 20;                   // Max number of terms
const double DOMAIN_MIN = 2.0;          // Min domain length
const double DOMAIN_MAX = 100.0;        // Max domain length
const double AMP_MIN = -10.0;           // Min amplitude
const double AMP_MAX = 10.0;            // Max amplitude
// ... (see source for complete list)
```

### Command Line Arguments

The generator supports the following command line arguments to override defaults:

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--samples <num>` | `-s` | Number of training samples to generate | 2 |
| `--append` | `-a` | Append to existing samples (flag, no value) | false |
| `--time-step <value>` | `-ts` | Time step size | 0.000001 |
| `--num-time-steps <num>` | `-nts` | Number of time steps | 100000 |

**Behavior:**
- **Without `--append`**: Deletes all existing `sample_*` folders and starts numbering from `sample_000000`
- **With `--append`**: Finds the highest existing sample number and continues from the next number (e.g., if `sample_000007` exists, new samples start at `sample_000008`)

### Running the Generator

```bash
cd build

# Generate 10 samples with default settings (fresh start)
./training_data_generator --samples 10

# Generate 5 more samples, appending to existing dataset
./training_data_generator --samples 5 --append

# Custom time configuration
./training_data_generator -s 20 -ts 0.00001 -nts 50000

# Append with custom time settings
./training_data_generator -s 3 -a -ts 0.000001 -nts 100000

# Run with defaults (2 samples, fresh start)
./training_data_generator
```

**Error Handling:**
The generator validates all command line arguments and will exit with an error message if:
- An unknown argument is provided
- A required value is missing (e.g., `--samples` without a number)
- An invalid value is provided (e.g., `--samples abc`)

### Output Structure

This creates:
```
training_data/
├── sample_000000/
│   ├── solution.bin
│   └── metadata.json          # IC parameters and solver config
├── sample_000001/
│   └── ...
└── ...
```

Each `metadata.json` contains:
- **Configuration parameters**: n, domain_length, amplitude ranges, frequency ranges, etc.
- **Exact terms**: amplitude, frequency, phase_shift for each sinusoid
- **Bias value**: vertical shift (if alwaysPositive mode used)
- **Solver parameters**: time_steps, time_step_size, num_domain_points, spatial_step_size, scheme_name

Example metadata structure:
```json
{
  "bias": 0.0,
  "config": {
    "n": 5,
    "domain_length": 10.0,
    "amp_min": -1.0,
    "amp_max": 1.0,
    ...
  },
  "solver": {
    "domain_length": 10.0,
    "kinematic_viscosity": 0.01,
    "num_domain_points": 1000,
    "scheme_name": "LaxWendroff",
    "spatial_step_size": 0.01,
    "time_step_size": 0.000001,
    "time_steps": 100000
  },
  "terms": [
    {
      "amplitude": 0.8,
      "frequency": 1.5,
      "phase_shift": 2.3
    },
    ...
  ]
}
```

## How the Solver Works

### Burgers' Equation

```
∂u/∂t + u(∂u/∂x) = ν(∂²u/∂x²)
```

Where:
- `u(x,t)`: velocity field
- `ν`: kinematic viscosity
- `∂u/∂t`: temporal change
- `u(∂u/∂x)`: convective (nonlinear) term
- `ν(∂²u/∂x²)`: diffusive term

### FTCS Discretization

```
u_i^(n+1) = u_i^n 
          - u_i^n * (Δt/Δx) * (u_(i+1)^n - u_(i-1)^n) / 2
          + (ν + ν_artificial) * (Δt/Δx²) * (u_(i+1)^n - 2*u_i^n + u_(i-1)^n)
```

### Lax-Wendroff Discretization

Uses a two-step approach with flux-based formulation:
- Predictor step computes intermediate fluxes
- Corrector step updates solution
- Second-order accurate in both space and time

### Boundary Conditions

Both schemes implement **periodic boundary conditions**:
```cpp
u[0] = u[N-1]  // Left boundary = right boundary
```

This is appropriate for:
- Wave propagation problems
- Fourier-based initial conditions
- Problems without physical boundaries

### Artificial Viscosity

Applied in compression regions to stabilize shocks:

```cpp
double ux = (u[i+1] - u[i-1]) / (2 * dx);
if (ux < 0) {  // Compression
    artificial_viscosity = cq * dx² * |ux|;
}
```

The `cq` parameter (typically 1-3) controls the amount of artificial dissipation.

## Visualization

The interactive visualizer allows you to play through simulation timesteps:

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Visualize data
python visualizer/visualize.py <folder_name> --speed 1

# Examples:
python visualizer/visualize.py data/step_function --speed 2
python visualizer/visualize.py training_data/sample_000000 --speed 0.5
```

**Controls:**
- **Play/Pause**: Start/stop animation
- **Prev/Next**: Step through frames manually
- **Speed Slider**: Adjust playback speed (0.1 - 50 frames/tick)
- **Lock Y Checkbox**: Toggle y-axis autoscaling
  - Unchecked: Y-axis scales to current frame
  - Checked: Y-axis locked to current limits

The visualizer searches for data in `training_data/<folder_name>` by default.

## Output Structure

### Experiment Data (`data/`)

```
data/
└── [run_name]/
    ├── solution.bin
    └── metadata.json
```

### Training Data (`training_data/`)

```
training_data/
└── sample_XXXXXX/
    ├── solution.bin
    └── metadata.json
```

### Binary Solution Format

- Simulation results are stored in `solution.bin`.

- The file contains a contiguous array of double values:

  [num_timesteps × num_domain_points]

- Data is written sequentially by timestep.
Each timestep contains `num_domain_points` velocity values.

- Metadata needed to interpret the binary file
(e.g., num_timesteps, num_domain_points, dx)
is stored in `metadata.json`.

### Metadata JSON Format

```json
{
  "bias": 0.0,
  "config": {
    "n": 5,
    "domain_length": 10.0,
    "amp_min": -1.0,
    "amp_max": 1.0,
    ...
  },
  "solver": {
    "domain_length": 10.0,
    "kinematic_viscosity": 0.01,
    "num_domain_points": 1000,
    "scheme_name": "LaxWendroff",
    "spatial_step_size": 0.01,
    "time_step_size": 0.000001,
    "time_steps": 100000
  },
  "terms": [
    {
      "amplitude": 0.8,
      "frequency": 1.5,
      "phase_shift": 2.3
    },
    ...
  ]
}
```

## Example Simulations

The `main.cpp` includes three demonstrations:

### 1. Step Function (Lax-Wendroff)
- Initial condition: velocity jump from 1 to 2 in [0.5, 1.0]
- Shows shock formation and diffusion
- Domain: [0, 2]

### 2. Sine Wave (Lax-Wendroff)
- Initial condition: sin(x) over [0, 2π]
- Demonstrates wave steepening and breaking
- Domain: [0, 2π]

### 3. Random Function (FTCS)
- Superposition of 5 random sinusoids
- Tests solver with complex initial conditions
- Domain: [0, 10]

## Stability Considerations

### CFL Condition

For stability, the Courant-Friedrichs-Lewy (CFL) condition must be satisfied:

```
CFL = u_max * Δt / Δx ≤ ALPHA
```

The solver automatically computes `Δx` based on:
- `ALPHA = 0.9` (safety factor)
- Maximum velocity in initial condition
- Time step size

### Choosing Parameters

**For smooth solutions:**
- FTCS is sufficient and faster
- Use moderate viscosity (ν ≈ 0.01)
- Smaller `cq` (1.0 - 2.0)

**For shock problems:**
- Lax-Wendroff captures shocks better
- May need higher artificial viscosity
- Larger `cq` (2.0 - 3.0)

**For training data:**
- Use consistent solver settings across samples
- Vary initial conditions broadly
- Save frequently to capture dynamics

## Notes and Best Practices

- **Memory usage**: The solver stores complete solution history in memory. For long simulations with fine grids, consider saving incrementally or using the `gap` parameter.
  
- **Spatial resolution**: Currently hardcoded to `dx = 0.01` (see comment in `burgers.cpp`). The automatic calculation based on CFL is disabled. Modify this for your specific needs.

- **Periodic boundaries**: Suitable for wave problems but not for problems requiring inflow/outflow conditions.

- **Random seed**: Use fixed seeds in `RandomInitialCondition` for reproducibility in training data.

- **Visualizer data location**: By default searches `training_data/`. Modify `run_visualizer()` in `visualize.py` if using `data/` directory.

- **Artificial viscosity**: Essential for capturing shocks without oscillations. Tune `cq` parameter based on problem characteristics.

## Extending the Solver

### Adding New schemes

1. Create a new class inheriting from `Burgerscheme` in `burger_scheme.h`
2. Implement `calculateNextU()` and `calculateArtificialViscosity()`
3. Add implementation file in `src/schemes/`
4. Update `CMakeLists.txt` to include new source file

Example:
```cpp
class Myscheme : public Burgerscheme {
public:
    void calculateNextU(...) override;
protected:
    double calculateArtificialViscosity(...) const override;
};
```

### Custom Initial Conditions

Implement any callable with signature `double(double)`:
```cpp
auto custom_ic = [](double x) -> double {
    return /* your function */;
};
```


## Notes

- The solver uses periodic boundary conditions (suitable for wave problems)
- Stability depends on the CFL condition: ensure time steps are small enough
- For large viscosity or long time periods, you may need more timesteps
- The solver stores the complete history in memory (consider memory usage for large simulations)

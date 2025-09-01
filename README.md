# DiffABM.jl: Differentiable Agent-Based Models in Julia

DiffABM.jl is a Julia package that implements differentiable agent-based models, enabling gradient-based optimization and machine learning applications for complex systems. The package provides implementations of classic ABM models including epidemiological (SIR), ecological (Sugarscape), and economic (Axtell firms) models, all designed to be automatically differentiable through Julia's automatic differentiation ecosystem.

## Features

- **Differentiable Agent-Based Models**: All models support automatic differentiation for gradient-based optimization and parameter inference
- **Classic Model Implementations**: 
  - **SIR Epidemiological Model**: Susceptible-Infected-Recovered dynamics with policy interventions
  - **Sugarscape**: Ecological model with resource competition and agent movement
  - **Axtell Firms**: Economic model of firm dynamics and worker allocation
- **GPU Acceleration**: Models are compatible with GPU computation through Flux.jl
- **Stochastic Differentiation**: Integration with StochasticAD.jl for handling stochastic processes
- **Flexible Architecture**: Easy to extend with custom agent behaviors and model components

## Installation

DiffABM.jl requires Julia 1.10 or later. Install the package using Julia's package manager:

```julia
using Pkg
Pkg.add("DiffABM")
```

For development or to run the latest version from source:

```julia
using Pkg
Pkg.add(url="https://github.com/arnauqb/DiffABM.jl")
```

## Quick Start

### SIR Epidemiological Model

```julia
using DiffABM

# Define model parameters
sir_params = SIRNonVecParams(
    neighbours=generate_network(n_agents=1000),
    model_params=SIRModelParams(
        beta=[0.3],    # infection rate
        gamma=[0.1],   # recovery rate
        quarantine_start_time=[10.0],
        quarantine_end_time=[50.0],
        p_quarantine=[0.5]
    ),
    n_agents=1000,
    delta_t=0.1,
    n_timesteps=100,
    discrete_sampler=GumbelSoftmax(temperature=0.1)
)

# Run simulation
results = abm_run(sir_params)
```

### Sugarscape Model

```julia
using DiffABM

# Initialize sugarscape environment
board_init = TwoPeakBoard(
    N=50,
    sugar_peaks=[15.0, 15.0, 35.0, 35.0],  # two peaks at (15,15) and (35,35)
    max_sugar=[4.0, 4.0],
    distance_function=euclidean_distance
)

agent_init = RandomAgentInitializer(
    n_agents=100,
    vision_range=3,
    metabolic_rate_range=(1.0, 4.0)
)

sugarscape_params = SugarScapeParams(
    board_initializer=board_init,
    agent_initializer=agent_init,
    n_timesteps=200
)

# Run simulation
results = abm_run(sugarscape_params)
```

### Axtell Firms Model

```julia
using DiffABM

# Define economic model parameters
axtell_params = AxtellFirmsParams(
    n_agents=500,
    n_firms=50,
    agent_initializer=RandomAxtellAgentInitializer(),
    utility_function=CobbDouglasUtility(),
    n_timesteps=100
)

# Run simulation
results = abm_run(axtell_params)
```

## Differentiable Modeling

All models support automatic differentiation, enabling gradient-based parameter optimization:

```julia
using Zygote, Optim

# Define loss function
function loss(params)
    results = abm_run(params)
    target_data = load_empirical_data()
    return sum((results.timeseries .- target_data).^2)
end

# Compute gradients
gradients = gradient(loss, sir_params)

# Optimize parameters
result = optimize(loss, initial_params, BFGS())
```

## Model Architecture

The package follows a modular design where each model implements:

- `abm_step(params)`: Single timestep evolution
- `abm_run(params)`: Full simulation execution
- `abm_logpdf(params)`: Log probability for Bayesian inference

All models leverage Julia's multiple dispatch system and are compatible with automatic differentiation frameworks including Zygote.jl, ForwardDiff.jl, and StochasticAD.jl.

## Dependencies

Key dependencies include:
- **Automatic Differentiation**: Zygote.jl, ForwardDiff.jl, ChainRulesCore.jl
- **Stochastic AD**: StochasticAD.jl
- **Neural Networks**: Flux.jl, GraphNeuralNetworks.jl  
- **Distributions**: Distributions.jl, DistributionsAD.jl
- **Discrete Sampling**: GumbelSoftmax.jl

## Testing

Run the test suite to verify installation:

```julia
using Pkg
Pkg.test("DiffABM")
```

Or run tests with the project environment:

```bash
julia --project -e "using Pkg; Pkg.test()"
```

## Contributing

Contributions are welcome! Please see the documentation for development guidelines and examples of extending the framework with custom models.

## Citation

If you use DiffABM.jl in your research, please cite:

```bibtex
@software{diffabm2024,
  title={DiffABM.jl: Differentiable Agent-Based Models in Julia},
  author={Quera-Bofarull, Arnau},
  year={2024},
  url={https://github.com/arnauqb/DiffABM.jl}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

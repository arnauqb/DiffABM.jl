# DiffABM.jl: Differentiable Agent-Based Models in Julia

> [!CAUTION]
> This is still very much an experimenting package

This package includes differentiable implementations of 3 popular agent-based models:

1. Axtell Model of Firms
2. Sugarscape
3. Network-based SIR model

## Installation

To install the package locally, simply run

```bash
julia --project
```

```julia
]instantiate
```

## Examples

You can find examples on how to run and differentiate each model in the `scripts` folder.

## TODO


## Citation

Please consider citing this work 

```bibtex
@misc{querabofarull2025automaticdifferentiationagentbasedmodels,
      title={Automatic Differentiation of Agent-Based Models}, 
      author={Arnau Quera-Bofarull and Nicholas Bishop and Joel Dyer and Daniel Jarne Ornia and Anisoara Calinescu and Doyne Farmer and Michael Wooldridge},
      year={2025},
      eprint={2509.03303},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2509.03303}, 
}
```
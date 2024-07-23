module DiffABM

import Distributions
using DiffResults
using Distributions
using ForwardDiff
using Flux
using GraphNeuralNetworks
using OffsetArrays
using Random
using StaticArrays
using StochasticAD

export abm_step, abm_run

abm_run(params) = throw("Not implemented for $(typeof(params))")
abm_step(params) = throw("Not implemented for $(typeof(params))")
abm_logpdf(params) = throw("Not implemented for $(typeof(params))")

# Write your package code here.
include("discrete_randomness.jl")

# models
include("BrockHommes.jl")
include("GameOfLife.jl")
include("SIR/sir.jl")

end

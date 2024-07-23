module DiffABM

import Distributions
using DiffResults
using ForwardDiff
using Flux
using GraphNeuralNetworks
using OffsetArrays
using StaticArrays
using StochasticAD

export abm_step, abm_run

abm_run(params) = throw("Not implemented for $(typeof(params))")
abm_step(params) = throw("Not implemented for $(typeof(params))")
abm_logpdf(params) = throw("Not implemented for $(typeof(params))")

# Write your package code here.
include("discrete_randomness.jl")
include("SIR/sir.jl")
include("GameOfLife.jl")

end

module DiffABM

using ChainRulesCore
import Distributions
using DiffResults
using Distributions
using ForwardDiff
using Functors
using Flux
using GraphNeuralNetworks
using GumbelSoftmax
using OffsetArrays
using Random
using SpecialFunctions
using StaticArrays
using StochasticAD

export abm_step, abm_run

abm_run(params) = throw("Not implemented for $(typeof(params))")
abm_step(params) = throw("Not implemented for $(typeof(params))")
abm_logpdf(params) = throw("Not implemented for $(typeof(params))")

# Write your package code here.
include("diff_utils.jl")
include("discrete_randomness.jl")

# models
include("BrockHommes.jl")
include("GameOfLife.jl")
include("RandomWalk.jl")
include("SIR/sir.jl")

end

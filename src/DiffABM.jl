module DiffABM

using ChainRulesCore
import Distributions
using DiffResults
using Distributions
using DistributionsAD
using DataStructures
using ForwardDiff
using Functors
using Flux
using GraphNeuralNetworks
using GumbelSoftmax
using LinearAlgebra
using Random
using SpecialFunctions
using StatsBase
using StochasticAD

export abm_step, abm_run

abm_run(params) = throw("Not implemented for $(typeof(params))")
abm_step(params) = throw("Not implemented for $(typeof(params))")
abm_logpdf(params) = throw("Not implemented for $(typeof(params))")

# Utils
include("utils.jl")
include("diff_utils.jl")
include("discrete_randomness.jl")

# models
include("random_walk.jl")
include("sugar_scape.jl")
include("axtell_firms.jl")
include("sir.jl")

end

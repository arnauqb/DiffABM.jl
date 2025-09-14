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
using OffsetArrays
using Random
using SpecialFunctions
using StatsBase
using StaticArrays
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
include("BrockHommes.jl")
include("RandomWalk.jl")
include("SugarScape.jl")
include("AxtellFirms.jl")
include("SIR.jl")

end

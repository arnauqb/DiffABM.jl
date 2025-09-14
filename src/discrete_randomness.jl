export GS, RGS, SAD, SM, ST, sample_categorical, sample_bernoulli
export DifferentiableBernoulli, DifferentiableOneHotCategorical

abstract type DiscreteSampler end
# GumbelSoftmax
struct GS <: DiscreteSampler
    tau::Float64
end
GS() = GS(0.1)
# Rao-GumbelSoftmax
struct RGS <: DiscreteSampler
    k::Int64
    tau::Float64
end
# StochasticAD
struct SAD{T} <: DiscreteSampler
    derivative_coupling::T
end
SAD() = SAD(nothing)
# StochasticAD with smoothing
struct SM <: DiscreteSampler end
# Straight-Through
struct ST <: DiscreteSampler end

function sample_categorical(::Union{SAD,SM}, probs)
    return [rand(Distributions.Categorical(probs[:, i])) for i in axes(probs, 2)]
end

function sample_categorical(::ST, probs)
    throw("Not implemented")
end


function sample_categorical(categorical_sampler::GS, probs::AbstractVector)
    categories = collect(1:size(probs, 2))
    one_hot = sample_gumbel_softmax(probs=probs, tau=categorical_sampler.tau)
    return sum(one_hot .* categories', dims=2)
end

function sample_bernoulli(::Union{SAD, SM}, p::Real)
    return rand(Distributions.Bernoulli(p))
end

function sample_bernoulli(sampler::Union{SAD,SM}, probs::AbstractVector)
    return sample_bernoulli.(Ref(sampler), probs)
end

function sample_bernoulli(bernoulli_sampler::GS, p::Real)
    probs_cat = [p 1.0 .- p]
    return sample_gumbel_softmax(
        probs=probs_cat, tau=bernoulli_sampler.tau, epsilon=1e-8)[1]
end

function sample_bernoulli(bernoulli_sampler::GS, probs::AbstractVector)
    probs_cat = [probs 1.0 .- probs]
    return sample_gumbel_softmax(
        probs=probs_cat, tau=bernoulli_sampler.tau, epsilon=1e-8)[:, 1]
end

function sample_bernoulli(bernoulli_sampler::RGS, probs::AbstractVector)
    probs_cat = [probs 1.0 .- probs]
    return sample_rao_gumbel_softmax(probs=probs_cat, tau=bernoulli_sampler.tau,
        k=bernoulli_sampler.k, epsilon=1e-8)[:, 1]
end

function sample_bernoulli(::ST, p::Real)
    return (rand() < p) + (p - ignore_gradient(p))
end

function sample_bernoulli(::ST, p::AbstractVector)
    return sample_bernoulli.(Ref(ST()), p)
end

function sample_categorical_onehot(probs)
    index = rand(Categorical(probs))
    return onehot(index, 1:length(probs))
end


struct DifferentiableBernoulli{R} <: Distributions.DiscreteUnivariateDistribution
    p::Float64
    rep_method::R
end
Base.length(d::DifferentiableBernoulli) = 1
function Distributions.rand(rng::AbstractRNG, d::DifferentiableBernoulli{R}) where {R<:Union{SM,SAD}}
    return rand(Bernoulli(d.p))
end
function Distributions.rand(rng::AbstractRNG, d::DifferentiableBernoulli{R}) where {R<:Union{GS,RGS}}
    probs = [p]
    probs_cat = [p 1.0 .- probs]
    sample = sample_gumbel_softmax(probs=probs_cat, tau=d.rep_method.tau, epsilon=1e-8)
    return sample[:, 1]
end
function Distributions.rand(rng::AbstractRNG, d::DifferentiableBernoulli{ST})
    return (rand() < ignore_gradient(d.p)) + (d.p - ignore_gradient(d.p))
end

function Distributions.logpdf(d::DifferentiableBernoulli, x)
    return logpdf(Bernoulli(d.p), x)
end

"""
One-hot categorical distribution with differentiable sampling. For each gradient estimator,
we need to define a different transformation for the gradient passs. This is because while
the Straight-Through and Gumbel-Softmax estimators naturally output a one-hot vector,
StochasticAD returns an index which needs to be converted to a one-hot vector in a differentiable way.
"""
struct DifferentiableOneHotCategorical{T,R} <: Distributions.DiscreteMultivariateDistribution
    probs::Vector{T}
    rep_method::R
end
Base.length(d::DifferentiableOneHotCategorical) = length(d.probs)
function Distributions.rand(rng::AbstractRNG, d::DifferentiableOneHotCategorical{T,R}) where {T,R<:Union{SM}}
    # NOTE: not clear if this works
    probs = d.probs ./ sum(d.probs)
    index = rand(Categorical(probs))
    index_onehot_hard = Flux.onehot(ignore_gradient(index), 1:length(d.probs))
    positions = 1:length(d.probs)
    distances = abs.(positions .- index)
    k = 1.0
    index_onehot_soft = my_softmax(k .* distances)
    index_onehot = index_onehot_hard + (index_onehot_soft - ignore_gradient.(index_onehot_soft))
    return index_onehot
end
function Distributions.rand(rng::AbstractRNG, d::DifferentiableOneHotCategorical{T,R}) where {T,R<:SAD}
    # NOTE: not clear if this works
    probs = d.probs ./ sum(d.probs)
    index = rand(Categorical(probs))
    onehot_matrix = Diagonal(ones(length(d.probs)))
    return onehot_matrix[index, :]
end
function Distributions.rand(rng::AbstractRNG, d::DifferentiableOneHotCategorical{T,R}) where {T,R<:Union{GS,RGS}}
    probs = reshape(d.probs, 1, :)
    probs = probs ./ sum(probs)
    return sample_gumbel_softmax(probs=probs, tau=d.rep_method.tau, epsilon=1e-8)[1, :]
end
function Distributions.rand(rng::AbstractRNG, d::DifferentiableOneHotCategorical{T,R}) where {T,R<:ST}
    probs = d.probs ./ sum(d.probs)
    index = rand(Categorical(ignore_gradient.(probs)))
    index_onehot_hard = Flux.onehot(ignore_gradient(index), 1:length(probs))
    return index_onehot_hard + (probs - ignore_gradient.(probs))
end
function Distributions.rand!(rng::AbstractRNG, d::DifferentiableOneHotCategorical, x::AbstractArray{<:Real,M}) where {M}
    x .= rand(rng, d)
end
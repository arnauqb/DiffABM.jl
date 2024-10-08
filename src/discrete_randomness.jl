export GS, RGS, SAD, SM, ST, sample_categorical, sample_bernoulli

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
SAD() = SAD(StochasticAD.InversionMethodDerivativeCoupling())
# StochasticAD with smoothing
struct SM <: DiscreteSampler end
# Straight-Through
struct ST <: DiscreteSampler end

function sample_categorical(::Union{SAD, SM}, probs)
    return [rand(Distributions.Categorical(probs[:, i])) for i in axes(probs, 2)]
end

function sample_categorical(::ST, probs)
    res = [rand(Distributions.Categorical(probs[:, i])) for i in axes(probs, 2)]
    return ignore_gradient(res) + (probs - ignore_gradient(probs))
end

function sample_categorical(categorical_sampler::GS, probs)
    categories = collect(1:size(probs, 2))
    one_hot = sample_gumbel_softmax(probs = probs, tau = categorical_sampler.tau)
    return sum(one_hot .* categories', dims = 2)
end

function sample_bernoulli(::Union{SAD, SM}, probs)
    return [rand(Distributions.Bernoulli(p)) for p in probs]
end

function sample_bernoulli(bernoulli_sampler::GS, probs)
    probs_cat = [probs 1.0 .- probs]
    return sample_gumbel_softmax(
        probs = probs_cat, tau = bernoulli_sampler.tau, epsilon = 1e-8)[:, 1]
end

function sample_bernoulli(bernoulli_sampler::RGS, probs)
    probs_cat = [probs 1.0 .- probs]
    return sample_rao_gumbel_softmax(probs = probs_cat, tau = bernoulli_sampler.tau,
        k = bernoulli_sampler.k, epsilon = 1e-8)[:, 1]
end

# forward diff rule
function sample_bernoulli(::ST, p)
    return (rand() < p) + (p - ignore_gradient(p))
end

function sample_bernoulli(::ST, p::AbstractVector)
    return sample_bernoulli.(Ref(ST()), p)
end

function sample_categorical_onehot(probs)
    index = rand(Categorical(probs))
    return onehot(index, 1:length(probs))
end
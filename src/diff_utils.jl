export SigmoidSmoothing, GaussianSmoothing, PiecewiseSmoothing, StraightThroughSmoothing,
       TwoPiecewiseSmoothing, DiffBeta
ignore_gradient(x) = ChainRulesCore.ignore_derivatives(x)
ignore_gradient(x::ForwardDiff.Dual) = ForwardDiff.value(x)
ignore_gradient(x::StochasticAD.StochasticTriple) = StochasticAD.value(x)

function my_sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end

abstract type StepSmoothing end

struct GaussianSmoothing{T} <: StepSmoothing
    σ::T
end
function (smoothing::GaussianSmoothing)(x)
    return 0.5 * (1.0 + erf(x / (smoothing.σ * sqrt(2.0))))
end

struct SigmoidSmoothing{T} <: StepSmoothing
    k::T
end
function (smoothing::SigmoidSmoothing)(x)
    return 1.0 / (1.0 + exp(-smoothing.k * x))
end

struct StraightThroughSmoothing <: StepSmoothing end
function (smoothing::StraightThroughSmoothing)(x)
    return x
end

struct PiecewiseSmoothing <: StepSmoothing
    thresholds::Vector{Float64}
end
function (smoothing::PiecewiseSmoothing)(x)
    if ignore_gradient(x) < smoothing.thresholds[1]
        return 0.0
    elseif ignore_gradient(x) > smoothing.thresholds[2]
        return 1.0
    else
        return (x - smoothing.thresholds[1]) /
               (smoothing.thresholds[2] - smoothing.thresholds[1])
    end
end

struct TwoPiecewiseSmoothing <: StepSmoothing
    xs::Vector{Float64}
    y::Float64
end
function (smoothing::TwoPiecewiseSmoothing)(x)
    a, b = smoothing.xs
    y = smoothing.y
    if ignore_gradient(x) < -a
        return 0.0
    elseif ignore_gradient(x) < -b
        return y * (x + a) / (a - b)
    elseif ignore_gradient(x) < b
        m = (1 - 2y) / (2b)
        n = 0.5
        return m * x + n
    elseif ignore_gradient(x) < a
        m = -y / (b - a)
        n = (1 - y) - (b * m)
        return m * x + n
    else
        return 1.0
    end
end
## Differentiable step
function soft_step(smoothing::StepSmoothing, threshold, x)
    return smoothing(x - threshold)
end

function differentiable_step(smoothing::StepSmoothing, threshold, x::Real)
    soft = soft_step(smoothing, threshold, x)
    hard = (x > threshold)
    return hard + (soft - ignore_gradient(soft))
end

## Differentiable gate
function hard_gate(a, b, x)
    return x > a && x < b ? 1.0 : 0.0
end

function soft_gate(smoothing::StepSmoothing, a, b, x)
    return smoothing(x - a) * smoothing(b - x)
end

function differentiable_gate(smoothing::StepSmoothing, a, b, x::Real)
    soft = soft_gate(smoothing, a, b, x)
    return hard_gate(a, b, x) + (soft - ignore_gradient(soft))
end

function my_softmax(array; k=1.0)
    return exp.(k .* array) ./ sum(exp.(k .* array))
end

## Differentiable argmax
function random_argmax(array)
    max_value = maximum(array)
    max_indices = findall(x -> x == max_value, array)
    return rand(max_indices)
end
function differentiable_argmax(array::Vector{<:StochasticAD.StochasticTriple})
    # shift by maximum to avoid overflow
    soft = my_softmax(array .- maximum(array))
    hard = random_argmax(ignore_gradient.(array))
    hard_onehot = Flux.onehot(hard, 1:length(array))
    return hard_onehot + (soft - ignore_gradient.(soft))
end
function differentiable_argmax(array)
    # shift by maximum to avoid overflow and perturbate to avoid ties
    soft = Flux.softmax(array .- maximum(array))
    hard = random_argmax(ignore_gradient.(array))
    hard_onehot = Flux.onehot(hard, 1:length(array))
    return hard_onehot + (soft - ignore_gradient.(soft))
end

## Differentiable is less and is greater

function differentiable_is_greater(smoothing::StepSmoothing, x, threshold)
    return differentiable_step(smoothing, threshold, x)
end

function differentiable_is_less(smoothing::StepSmoothing, x, threshold)
    return 1.0 - differentiable_is_greater(smoothing, x, threshold)
end

## logic
function differentiable_and(x, y)
    return x * y
end

function differentiable_or(x, y)
    return x + y - x * y
end

function differentiable_not(x)
    return one(typeof(x)) - x
end

## differentiable indexing

function soft_index(smoothing::StepSmoothing, array, index)
    # for each element in array, deffine a differentiable gate.
    n = length(array)
    ret = zero(array[1])
    for i in 1:n
        gate = differentiable_gate(smoothing, i, i + 1, index)
        ret += array[i] .* gate
    end
    return ret
end

function differentiable_index(smoothing::StepSmoothing, array, index)
    soft = soft_index(smoothing, array, index)
    hard = array[Int64(ignore_gradient(index))]
    return @. hard + (soft - ignore_gradient(soft))
end

function soft_index_to_onehot(array, index)
end

## soft histogram
function soft_histogram(samples, bin_centers; σ = 1.0)
    N = length(samples)
    M = length(bin_centers)
    sigmas = diff(vcat(0.0, bin_centers)) ./ 2.0

    # Compute the soft histogram
    hist = zeros(M)
    for i in 1:M
        hist[i] = sum(exp.(-(samples .- bin_centers[i]) .^ 2 ./ (2sigmas[i]^2))) /
                  (N * sigmas[i] * sqrt(2π))
    end

    return hist
end

function hard_histogram(samples, n_agents)
    edges = 10 .^ range(log10(1.0), log10(n_agents), length=100)
    h = StatsBase.fit(StatsBase.Histogram, samples, edges, closed=:left)
    return h
end

function histogram(samples, n_agents)
    h = hard_histogram(samples, n_agents)
    edges = collect(h.edges[1])
    bin_centers = edges[1:end-1] + diff(edges) ./ 2.0
    soft_h = soft_histogram(samples, bin_centers)
    return bin_centers, h.weights / length(samples), soft_h
end

function wasserstein_loss(hist1, hist2, bin_centers)
    # Ensure histograms have the same length
    @assert length(hist1) == length(hist2)

    # Compute cumulative distributions
    cdf1 = cumsum(hist1)
    cdf2 = cumsum(hist2)

    # Normalize CDFs
    cdf1 = cdf1 ./ cdf1[end]
    cdf2 = cdf2 ./ cdf2[end]

    # Compute bin widths
    bin_widths = diff(vcat(2bin_centers[1] - bin_centers[2], bin_centers,
        2bin_centers[end] - bin_centers[end - 1])) / 2

    # Compute Wasserstein distance
    distance = sum(abs.(cdf1 .- cdf2) .* bin_widths)

    return distance
end

function kde(samples, evaluation_points; bandwidth=nothing)
    n = length(samples)
    
    # Silverman's rule of thumb for bandwidth selection if not provided
    if isnothing(bandwidth)
        σ = std(samples)
        bandwidth = 1.06 * σ * n^(-1/5) .* ones(length(evaluation_points))
    end
    
    # Compute KDE
    density = zeros(length(evaluation_points))
    for (i, x) in enumerate(evaluation_points)
        kernel_sum = sum(pdf.(Normal(0, bandwidth[i]), x .- samples))
        density[i] = kernel_sum / (n * bandwidth[i])
    end
    
    return density
end

Base.isinteger(x::Real, tol::Float64) = abs(round(x) - x) < tol

struct DiffBeta{T} <: Distributions.ContinuousUnivariateDistribution
    alpha::T
    beta::T
end
function Distributions.rand(rng::AbstractRNG, dist::DiffBeta{T}, n_samples::Int64) where {T}
    # sample from Kumaraswamy distribution as a quick diff approximation
    u = rand(rng, n_samples)
    return (1 .- u .^ (1 / dist.beta)) .^ (1 / dist.alpha)
end
function Distributions.rand(rng::AbstractRNG, dist::DiffBeta{T}) where {T}
    u = rand(rng)
    return (1 .- u .^ (1 / dist.beta)) .^ (1 / dist.alpha)
end
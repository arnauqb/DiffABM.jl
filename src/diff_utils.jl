export SigmoidSmoothing, GaussianSmoothing, PiecewiseSmoothing, StraightThroughSmoothing
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
	return sigmoid(smoothing.k * x)
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
        return (x - smoothing.thresholds[1]) / (smoothing.thresholds[2] - smoothing.thresholds[1])
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

function my_softmax(array)
    return exp.(array) ./ sum(exp.(array))
end

## Differentiable argmax
function random_argmax(array)
    max_value = maximum(array)
    max_indices = findall(x -> x == max_value, array)
    return rand(max_indices)
end
function differentiable_argmax(array)
    soft = my_softmax(array .+ 1.0) # add 1.0 to avoid overflow
    hard = random_argmax(ignore_gradient.(array))
    hard_onehot = Flux.onehot(hard, 1:length(array))
    return hard_onehot + (soft - ignore_gradient.(soft))
end

## Differentiable is less and is greater

function differentiable_is_greater(smoothing::StepSmoothing, x, y)
    return differentiable_step(smoothing, y, x)
end

function differentiable_is_less(smoothing::StepSmoothing, x, y)
    return 1.0 - differentiable_is_greater(smoothing, x, y)
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

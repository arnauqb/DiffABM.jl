export SigmoidSmoothing, GaussianSmoothing
ignore_gradient(x) = ChainRulesCore.ignore_derivatives(x)
ignore_gradient(x::ForwardDiff.Dual) = ForwardDiff.value(x)
ignore_gradient(x::StochasticAD.StochasticTriple) = StochasticAD.value(x)


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

## Differentiable step
function soft_step(smoothing::StepSmoothing, a, x)
    return smoothing(x - a)
end

function differentiable_step(smoothing::StepSmoothing, a, x::Real)
    soft = soft_step(smoothing, a, x)
    hard = (x >= a)
    return hard + (soft - ignore_gradient(soft))
end


## Differentiable gate
function hard_gate(a, b, x)
	return x >= a && x <= b ? 1.0 : 0.0
end

function soft_gate(smoothing::StepSmoothing, a, b, x)
	return smoothing(x - a) * smoothing(b - x)
end

function differentiable_gate(smoothing::StepSmoothing, a, b, x::Real)
	soft = soft_gate(smoothing, a, b, x)
	return hard_gate(a, b, x) + (soft - ignore_gradient(soft))
end

## Differentiable argmax
function differentiable_argmax(array::AbstractArray{T}) where T <: Real
    soft = softmax(array)
    hard = argmax(array)
    hard_onehot = Flux.onehot(hard, 1:length(array))
    return hard_onehot + (soft - ignore_gradient.(soft))
end

## Differentiable is less and is greater

function differentiable_is_less(smoothing::StepSmoothing, x, y)
    return differentiable_step(smoothing, x, y)
end

function differentiable_is_greater(smoothing::StepSmoothing, x, y)
    return differentiable_step(smoothing, y, x)
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

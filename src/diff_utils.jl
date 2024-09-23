ignore_gradient(x) = ChainRulesCore.ignore_derivatives(x)
ignore_gradient(x::ForwardDiff.Dual) = ForwardDiff.value(x)
ignore_gradient(x::StochasticAD.StochasticTriple) = StochasticAD.value(x)

function hard_step(a, b, x)
	return x >= a && x <= b ? 1.0 : 0.0
end

function soft_step(a, b, x)
	return sigmoid(x - a) * sigmoid(b - x)
end

function gaussian_smoothing(x, sigma = 1.0)
	return 0.5 * (1.0 + erf(x / (sigma * sqrt(2.0))))
end

function soft_step_gaussian(a, b, x)
	return gaussian_smoothing(x - a) * gaussian_smoothing(b - x)
end

function differentiable_step(a, b, x)
	return hard_step(a, b, x) + (soft_step_gaussian(a, b, x) - ignore_gradient(soft_step_gaussian(a, b, x)))
end

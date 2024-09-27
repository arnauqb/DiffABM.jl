using DiffABM
using LinearAlgebra
using ForwardDiff

##
function f(index)
    # array of 1s in the diagional and upper diagonal
    matrix = [[i >=j ? 1 : 0 for j in 1:3] for i in 1:3]
    return DiffABM.differentiable_index(GaussianSmoothing(1.0), matrix, index)
end
f(2.0)
ForwardDiff.derivative(f, 2.0)


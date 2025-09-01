using Test
using DiffABM
using ForwardDiff

@testset "diff_utils" begin
    @testset "smoothing functions" begin
        gs = GaussianSmoothing(1.0)
        @test gs(-5.0) ≈ 0.0 atol=1e-3
        @test gs(0.0) ≈ 0.5
        @test gs(5.0) ≈ 1.0 atol=1e-3
        @test ForwardDiff.derivative(gs, 0.0) > 0  # positive derivative
        
        ss = SigmoidSmoothing(2.0)
        @test ss(-5.0) < 0.1
        @test ss(0.0) ≈ 0.5
        @test ss(5.0) > 0.9
        @test ForwardDiff.derivative(ss, 0.0) > 0  # positive derivative
    end
    
    @testset "differentiable step" begin
        smoothing = GaussianSmoothing(1.0)
        @test DiffABM.differentiable_step(smoothing, 1.0, 2.0) == 1.0
        @test DiffABM.differentiable_step(smoothing, 1.0, 0.5) == 0.0
        # Derivative should be small but not zero at middle of smoothing region
        @test abs(ForwardDiff.derivative(x -> DiffABM.differentiable_step(smoothing, 1.0, x), 1.5)) < 0.5
        @test ForwardDiff.derivative(x -> DiffABM.differentiable_step(smoothing, 1.0, x), 1.1) > 0
    end
    
    @testset "differentiable gate" begin
        smoothing = GaussianSmoothing(1.0)
        @test DiffABM.differentiable_gate(smoothing, 2.0, 5.0, 3.0) == 1.0
        @test DiffABM.differentiable_gate(smoothing, 2.0, 5.0, 1.0) == 0.0
        @test DiffABM.differentiable_gate(smoothing, 2.0, 5.0, 6.0) == 0.0
        # Test differentiability - derivative should be small in middle
        @test abs(ForwardDiff.derivative(x -> DiffABM.differentiable_gate(smoothing, 2.0, 5.0, x), 3.5)) < 0.5
        @test ForwardDiff.derivative(x -> DiffABM.differentiable_gate(smoothing, 2.0, 5.0, x), 2.1) > 0
    end
    
    @testset "logic functions" begin
        @test DiffABM.differentiable_and(0.8, 0.6) ≈ 0.48
        @test DiffABM.differentiable_or(0.4, 0.3) ≈ 0.58
        @test DiffABM.differentiable_not(0.7) ≈ 0.3
        # Test derivatives
        @test ForwardDiff.derivative(x -> DiffABM.differentiable_and(x, 0.6), 0.8) ≈ 0.6
        @test ForwardDiff.derivative(x -> DiffABM.differentiable_or(x, 0.3), 0.4) ≈ 0.7
        @test ForwardDiff.derivative(x -> DiffABM.differentiable_not(x), 0.7) ≈ -1.0
    end
    
    @testset "differentiable argmax" begin
        arr = [1.0, 3.0, 2.0]
        result = DiffABM.differentiable_argmax(arr)
        @test length(result) == 3
        @test sum(result) ≈ 1.0
        @test result[2] == 1.0  # max is at index 2
        # Test jacobian (argmax returns vector, so use jacobian not gradient)
        jac = ForwardDiff.jacobian(DiffABM.differentiable_argmax, arr)
        @test size(jac) == (3, 3)
        @test vec(sum(jac, dims=1)) ≈ [0.0, 0.0, 0.0] atol=1e-10  # columns sum to 0
    end
    
    @testset "differentiable indexing" begin
        smoothing = GaussianSmoothing(0.5)
        arr = [1.0, 4.0, 2.0]
        # Test exact indexing
        @test DiffABM.differentiable_index(smoothing, arr, 2.0) == 4.0
        # Test differentiability
        grad = ForwardDiff.gradient(x -> DiffABM.differentiable_index(smoothing, x, 2.0), arr)
        @test grad[2] == 1.0  # gradient w.r.t. selected element
        @test sum(abs.(grad[[1,3]])) < 0.1  # small gradients for unselected
    end

end
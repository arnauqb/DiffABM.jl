using Test
using DiffABM
using ForwardDiff

@testset "diff_utils" begin
    @testset "differentiable step" begin
        @test DiffABM.differentiable_step(0.0, 1.0, 0.5) == 1.0
        @test DiffABM.differentiable_step(0.0, 1.0, -0.5) == 0.0
        @test DiffABM.differentiable_step(0.0, 1.0, 1.5) == 0.0
        @test DiffABM.differentiable_step(0.0, 1.0, 0.2) == 1.0
        @test ForwardDiff.derivative(x -> DiffABM.differentiable_step(0.0, 1.0, x), 0.5) ≈ 0.0 atol = 1e-6
        @test ForwardDiff.derivative(x -> DiffABM.differentiable_step(0.0, 1.0, x), 10.5) ≈ 0.0 atol = 1e-6
        @test ForwardDiff.derivative(x -> DiffABM.differentiable_step(0.0, 1.0, x), -10.5) ≈ 0.0 atol = 1e-6
        @test ForwardDiff.derivative(x -> DiffABM.differentiable_step(0.0, 1.0, x), 0.2) > 0.0
        @test ForwardDiff.derivative(x -> DiffABM.differentiable_step(0.0, 1.0, x), 0.8) < 0.0
    end

end
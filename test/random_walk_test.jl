using DiffABM
using Test

@testset "test random walk" begin
    params = RandomWalkParams(100, SM(), [0.5])
    x = abm_run(params)
    @test length(x) == 100
end
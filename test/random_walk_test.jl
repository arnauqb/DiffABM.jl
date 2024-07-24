using DiffABM
using Test

@testset "test random walk" begin
    for discrete_sampler in [GS(0.1), SAD(), ST(), SM()]
        params = RandomWalkParams(100, discrete_sampler, [0.5])
        x = abm_run(params)
        @test length(x) == 100
    end
end
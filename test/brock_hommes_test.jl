using DiffABM
using Test

@testset "test Brock-Hommes model" begin
    p = [120, 0, 0.9, 0.9, 1.01, 0, 0.2, -0.2, 0.0, 0.04, 1.0]
    bh = BrockHommesParams(100, p)
    x = abm_run(bh)
    @test length(x) == 100
end
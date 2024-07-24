using DiffABM
using Test

@testset "test game of life" begin
    n_timesteps = 50
    probs = rand(10)
    N = 20
    initial_p = [0.5]
    for discrete_sampler in [GS(0.1), SAD(), ST(), SM()]
        gol_params = GameOfLifeParams(n_timesteps, N, probs, initial_p, discrete_sampler)
        board_history = abm_run(gol_params)
        @test board_history[1] / (2N)^2 ≈ initial_p[1] atol = 0.075
        @test length(board_history) == n_timesteps
    end
end;

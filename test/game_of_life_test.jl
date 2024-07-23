using DiffABM
using Test

@testset "test game of life" begin
    n_timesteps = 50
    birth_probs = rand(5)
    N = 10
    discrete_sampler = SAD()
    initial_p = 0.5
    gol_params = GameOfLifeParams(n_timesteps, N, birth_probs, initial_p, discrete_sampler)
    board_history = abm_run(gol_params)
    initial_life = 0
    for i in -N:N
        for j in -N:N
            initial_life += board_history[1][i, j]
            for k in 1:n_timesteps
                @test board_history[k][i, j] ∈ [0, 1]
            end
        end
    end
    @test initial_life / (2 * N)^2 ≈ initial_p rtol = 0.05
end;

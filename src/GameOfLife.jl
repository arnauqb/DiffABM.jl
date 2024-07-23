# implementation from https://github.com/gaurav-arya/StochasticAD.jl/blob/main/tutorials/game_of_life/core.jl
export GameOfLifeParams

struct GameOfLifeParams{T, Q, S}
    n_timesteps::Int64
    board_length::Int64
    all_probs::Vector{T}
    initial_prob::Q
    discrete_sampler::S
    function GameOfLifeParams(n_timesteps::Int64, board_length, birth_probs::Vector{T},
            initial_prob::Q, discrete_sampler::S) where {T, Q, S}
        theta = birth_probs
        birth_probs = SA[theta[1], theta[2], theta[3], theta[4], theta[5]] # 0, 1, 2, 3, 4 neighbours
        death_probs = SA[
            1 - theta[1], 1 - theta[2], 1 - theta[3], 1 - theta[4], 1 - theta[5]] # 0, 1, 2, 3, 4 neighbours
        all_probs = vcat(birth_probs, death_probs)
        return new{T, Q, S}(
            n_timesteps, board_length, all_probs, initial_prob, discrete_sampler)
    end
end

function select_prob(all_probs, index::T) where {T}
    # since index is a float we need to express the indexing operation all_probs[index] as 
    # a sum of masks products.
    @assert index <= length(all_probs)
    res = zero(T)
    n = length(all_probs)
    for i in 1:n
        prod = one(T)
        for j in 1:n
            if i == j
                continue
            end
            prod *= (index - j) / (i - j)
        end
        res += all_probs[i] * prod
    end
    return res
end

function update_state!(bernoulli_sampler, all_probs, N, board, board_old)
    for i in (-N):N
        for j in (-N):N
            neighbours = board_old[i + 1, j] + board_old[i - 1, j] + board_old[i, j - 1] +
                         board_old[i, j + 1]
            index = board[i, j] * 5 + neighbours + 1 # trick necessary because we do not have implementation support for stochastic triple not <: Real
            prob = select_prob(all_probs, index)
            b = sample_bernoulli(bernoulli_sampler, prob)[1] 
            board[i, j] += (1 - 2 * board[i, j]) * b
        end
    end
    return board
end

function abm_step(params::GameOfLifeParams, x, t)
    new_board = copy(x)
    update_state!(
        params.discrete_sampler, params.all_probs, params.board_length, new_board, x)
    return new_board
end

function abm_run(params::GameOfLifeParams{T, Q, S}) where {T, Q, S}
    N = params.board_length
    board = OffsetArray(zeros(Q, 2 * N + 3, 2 * N + 3), (-(N + 1)):(N + 1),
        (-(N + 1)):(N + 1)) # pad by 1
    for i in (-N):N
        for j in (-N):N
            board[i, j] = sample_bernoulli(params.discrete_sampler, [params.initial_prob])[1]
        end
    end
    history = [copy(board)]
    for t in 1:(params.n_timesteps)
        board = abm_step(params, board, t)
        push!(history, copy(board))
    end
    return history
end
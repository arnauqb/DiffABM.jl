# implementation from https://github.com/gaurav-arya/StochasticAD.jl/blob/main/tutorials/game_of_life/core.jl
export GameOfLifeParams

struct GameOfLifeParams{T, Q, S}
    n_timesteps::Int64
    board_length::Int64
    probs::Vector{T}
    initial_prob::Vector{Q}
    discrete_sampler::S
end
@functor GameOfLifeParams (initial_prob, probs,)

function select_prob(probs, index::T) where {T}
    # since index is a float we need to express the indexing operation all_probs[index] as 
    # a sum of masks products.
    @assert index <= length(probs)
    res = zero(T)
    n = length(probs)
    for i in 1:n
        prod = one(T)
        for j in 1:n
            if i == j
                continue
            end
            prod *= (index - j) / (i - j)
        end
        res += probs[i] * prod
    end
    return res
end

function update_state!(bernoulli_sampler, probs, N, board, board_old)
    for i in (-N):N
        for j in (-N):N
            neighbours = board_old[i + 1, j] + board_old[i - 1, j] + board_old[i, j - 1] +
                         board_old[i, j + 1]
            index = board[i, j] * 5 + neighbours + 1 # trick necessary because we do not have implementation support for stochastic triple not <: Real
            prob = select_prob(probs, index)
            b = sample_bernoulli(bernoulli_sampler, prob)[1] 
            board[i, j] += (1 - 2 * board[i, j]) * b
        end
    end
    return board
end

function abm_step(params::GameOfLifeParams, x, t)
    new_board = copy(x)
    update_state!(
        params.discrete_sampler, params.probs, params.board_length, new_board, x)
    return new_board
end

function abm_run(params::GameOfLifeParams{T, Q, S}) where {T, Q, S}
    N = params.board_length
    board = OffsetArray(zeros(Q, 2 * N + 3, 2 * N + 3), (-(N + 1)):(N + 1),
        (-(N + 1)):(N + 1)) # pad by 1
    for i in (-N):N
        for j in (-N):N
            board[i, j] = sample_bernoulli(params.discrete_sampler, params.initial_prob)[1]
        end
    end
    history = [copy(board)]
    for t in 2:(params.n_timesteps)
        board = abm_step(params, board, t)
        push!(history, copy(board))
    end
    return sum.(history)
end
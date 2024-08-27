export GameOfLifeParams

"""
    GameOfLifeParams{T, Q, S}

Parameters for the Game of Life simulation.

# Fields
- `n_timesteps::Int64`: Number of timesteps to simulate
- `board_length::Int64`: Length of one side of the square board
- `probs::Vector{T}`: Probabilities for cell state transitions
- `initial_prob::Vector{Q}`: Initial probabilities for cell states
- `discrete_sampler::S`: Sampler for discrete random variables
"""
struct GameOfLifeParams{T, Q, S}
    n_timesteps::Int64
    board_length::Int64
    probs::Vector{T}
    initial_prob::Vector{Q}
    discrete_sampler::S
end
@functor GameOfLifeParams (initial_prob, probs,)

"""
    select_prob(probs, index::T) where {T}

Select a probability from `probs` based on the given `index`.

This function interpolates between discrete probability values when `index` is not an integer.

# Arguments
- `probs`: Vector of probabilities
- `index::T`: Index to select probability (can be a float)

# Returns
- Selected probability value
"""
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

"""
    update_state!(bernoulli_sampler, probs, N, board, board_old)

Update the state of the Game of Life board for one timestep.

# Arguments
- `bernoulli_sampler`: Sampler for Bernoulli random variables
- `probs`: Vector of probabilities for cell state transitions
- `N`: Half the length of one side of the board
- `board`: Current state of the board (will be modified)
- `board_old`: Previous state of the board

# Returns
- Updated `board`
"""
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

"""
    abm_step(params::GameOfLifeParams, x, t)

Perform one step of the Game of Life simulation.

# Arguments
- `params::GameOfLifeParams`: Parameters for the simulation
- `x`: Current state of the board
- `t`: Current timestep (not used in this function)

# Returns
- New state of the board after one step
"""
function abm_step(params::GameOfLifeParams, x, t)
    new_board = copy(x)
    update_state!(
        params.discrete_sampler, params.probs, params.board_length, new_board, x)
    return new_board
end

"""
    abm_run(params::GameOfLifeParams{T, Q, S}) where {T, Q, S}

Run the full Game of Life simulation for the specified number of timesteps.

# Arguments
- `params::GameOfLifeParams{T, Q, S}`: Parameters for the simulation

# Returns
- Vector of board states at each timestep, where each state is represented by the sum of all cell values
"""
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
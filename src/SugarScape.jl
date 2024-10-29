export SugarScapeParams, TwoPeakBoard, RandomAgentInitializer, GeneratedAgentInitializer,
    ConstantAgentInitializer, VonNeumannNeighborhood, MooreNeighborhood, GeneratedBoard

using Base

struct SugarSeeker{T,V,M}
    id::Int64
    vision_matrix::V
    metabolic_rate::M
    i::Vector{Int64}
    j::Vector{Int64}
    x::Vector{T}
    y::Vector{T}
    alive::Vector{T}
    wealth::Vector{T}
end
SugarSeeker(; id, vision_matrix, metabolic_rate, i, j, x, y, alive, wealth) =
    SugarSeeker(id, vision_matrix, metabolic_rate, i, j, x, y, alive, wealth)

abstract type BoardInitializer end

struct GeneratedBoard{T} <: BoardInitializer
    N::Int64
    board::Vector{T}
end
get_type(board::GeneratedBoard) = typeof(board.board[1])
@functor GeneratedBoard (board,)

struct TwoPeakBoard{T,Q} <: BoardInitializer
    N::Int
    sugar_peaks::Vector{T}
    max_sugar::Vector{Q}
    distance_function::Function
end
function get_type(board::TwoPeakBoard)
    promote_type(typeof(board.max_sugar[1]), typeof(board.sugar_peaks[1]))
end
@functor TwoPeakBoard (sugar_peaks, max_sugar)

function initialize_board(initializer::TwoPeakBoard, diff_type)
    N, max_sugar = initializer.N, initializer.max_sugar
    board = zeros(diff_type, N, N)
    sugar_peak_xs = initializer.sugar_peaks[1:2:end]
    sugar_peak_ys = initializer.sugar_peaks[2:2:end]
    n_peaks = length(sugar_peak_xs)
    for i in 1:N
        for j in 1:N
            for k in 1:n_peaks
                dist = initializer.distance_function(
                    (i, j), (sugar_peak_xs[k], sugar_peak_ys[k])) + 1e-3
                sugar = max_sugar[1] * exp(-dist)
                board[i, j] += sugar
            end
        end
    end
    return board
end
function initialize_board(initializer::GeneratedBoard, diff_type)
    return reshape(
        copy(convert.(diff_type, initializer.board)), initializer.N, initializer.N)
end

abstract type AgentInitializer end
abstract type Neighborhood end
struct VonNeumannNeighborhood <: Neighborhood end
struct MooreNeighborhood <: Neighborhood end

function is_within_vision(::VonNeumannNeighborhood, i, j, max_vision, vision_radius)
    abs(i - max_vision - 1) + abs(j - max_vision - 1) <= vision_radius
end

function is_within_vision(::MooreNeighborhood, i, j, max_vision, vision_radius)
    max(abs(i - max_vision - 1), abs(j - max_vision - 1)) <= vision_radius
end

"""
    generate_vision_matrices(max_vision)

Returns a matrix of size (2 * max_vision+1, 2 * max_vision+1) with zeros everywhere,
except for the squares that are away from the center by a distance less than or equal to vision_radius.
vision_radius varies from 1 to max_vision.
"""
function generate_vision_matrices(neighborhood::Neighborhood, max_vision)
    vision_matrices = Matrix{Float64}[]
    N = 2 * max_vision + 1
    for vision_radius in 1:max_vision
        vision_matrix = zeros(N, N)
        for i in 1:N
            for j in 1:N
                if is_within_vision(neighborhood, i, j, max_vision, vision_radius)
                    vision_matrix[i, j] = 1
                end
            end
        end
        push!(vision_matrices, vision_matrix)
    end
    return vision_matrices
end
get_max_vision(vision_matrix) = (size(vision_matrix, 1) - 2) ÷ 2

struct RandomAgentInitializer{T,M,W,U,N,S} <: AgentInitializer
    vision_distribution_probs::Vector{T}
    metabolic_rate_bounds::Vector{M}
    wealth_bounds::Vector{W}
    position_distribution::U
    neighborhood::N
    discrete_sampler::S
end
function RandomAgentInitializer(
    board_length;
    vision_distribution_probs,
    metabolic_rate_bounds,
    wealth_bounds=[8, 10],
    position_distribution=Product(DiscreteUniform.(
        [1, 1], [board_length, board_length])),
    neighborhood=VonNeumannNeighborhood(),
    discrete_sampler=ST())
    return RandomAgentInitializer(
        vision_distribution_probs,
        metabolic_rate_bounds,
        wealth_bounds,
        position_distribution,
        neighborhood,
        discrete_sampler
    )
end
@functor RandomAgentInitializer (vision_distribution_probs, metabolic_rate_bounds, wealth_bounds)
function get_type(initializer::RandomAgentInitializer)
    typeof(initializer.vision_distribution_probs[1])
end
"""
	initialize_agents(initializer::RandomAgentInitializer{T, M, W, U, N, S}, n_agents, diff_type) where {T, M, W, U, N, S}

Initialize a list of agents for the Sugarscape simulation.

# Arguments
- `initializer::RandomAgentInitializer{T, M, W, U, N, S}`: An instance of `RandomAgentInitializer` containing the distributions for agent attributes.
- `n_agents`: The number of agents to initialize.
- `diff_type`: The type used for differentiation (e.g., `Float64`).

# Returns
- A list of `SugarSeeker` agents initialized with random attributes based on the provided distributions.
"""
function initialize_agents(initializer::RandomAgentInitializer{T,M,W,U,N,S},
    n_agents, diff_type) where {T,M,W,U,N,S}
    agents = SugarSeeker[]
    # correct numerical error by renormalizing
    vision_probs = initializer.vision_distribution_probs
    # try to enlarge
    #vision_probs = sigmoid.(5 * (vision_probs .- 0.5))
    #k = 5
    #if ignore_gradient.(vision_probs) != [0.9, 0.1]
    #    vision_probs = sigmoid.(k * (vision_probs .- [0.9, 0.1]))
    #end
    vision_probs = vision_probs ./ sum(vision_probs)
    max_vision = length(vision_probs)
    metabolic_rate_dist = Beta(initializer.metabolic_rate_bounds[1], initializer.metabolic_rate_bounds[2])
    wealth_dist = Beta(initializer.wealth_bounds[1], initializer.wealth_bounds[2])

    vision_matrices = generate_vision_matrices(initializer.neighborhood, 8)
    vision_matrices = vision_matrices[[1, end]]
    for i in 1:n_agents
        vision_onehot_index = rand(DifferentiableOneHotCategorical(vision_probs, initializer.discrete_sampler))
        vision_matrix = sum(vision_matrices .* vision_onehot_index)
        metabolic_rate = 3.0 + 2 * rand(metabolic_rate_dist)
        wealth = 10 * rand(wealth_dist)
        position = rand(initializer.position_distribution)
        agent = SugarSeeker(
            i,
            vision_matrix,
            metabolic_rate,
            [position[1]],
            [position[2]],
            [convert(diff_type, position[1])],
            [convert(diff_type, position[2])],
            [one(diff_type)],
            [wealth])
        push!(agents, agent)
    end
    return agents
end

struct GeneratedAgentInitializer{V,M,W,P,S,N} <: AgentInitializer
    vision_probabilities::Vector{V}
    metabolic_rates::Vector{M}
    wealths::Vector{W}
    positions::Vector{P}
    discrete_sampler::S
    neighborhood::N
end
GeneratedAgentInitializer(; vision_probabilities, metabolic_rates, wealths, positions, discrete_sampler, neighborhood) =
    GeneratedAgentInitializer(vision_probabilities, metabolic_rates, wealths, positions, discrete_sampler, neighborhood)
@functor GeneratedAgentInitializer (vision_probabilities, metabolic_rates, wealths)
function get_type(initializer::GeneratedAgentInitializer)
    promote_type(eltype(initializer.metabolic_rates), eltype(initializer.wealths), eltype(initializer.vision_probabilities))
end
function initialize_agents(initializer::GeneratedAgentInitializer, n_agents, diff_type)
    max_vision = length(initializer.vision_probabilities)
    vision_matrices = generate_vision_matrices(initializer.neighborhood, max_vision)
    vision_probs = initializer.vision_probabilities ./ sum(initializer.vision_probabilities)
    agents = SugarSeeker[]
    for i in 1:n_agents
        vision_onehot_index = rand(DifferentiableOneHotCategorical(vision_probs, initializer.discrete_sampler))
        vision_matrix = sum(vision_matrices .* vision_onehot_index)
        position = initializer.positions[i]
        push!(agents, SugarSeeker(
            id=i,
            vision_matrix=vision_matrix,
            metabolic_rate=initializer.metabolic_rates[i],
            i=[position[1]],
            j=[position[2]],
            x=[convert(diff_type, position[1])],
            y=[convert(diff_type, position[2])],
            alive=[one(diff_type)],
            wealth=[convert(diff_type, initializer.wealths[i])]))
    end
    return agents
end


function compute_move(board, agent, occupied)
    scores = eltype(board)[]
    max_vision = get_max_vision(agent.vision_matrix)
    N = size(board, 1)
    for (vision_i, di) in enumerate(-max_vision:max_vision)
        for (vision_j, dj) in enumerate(-max_vision:max_vision)
            i_wrapped, j_wrapped = wrap_index(N, agent.i[1] + di, agent.j[1] + dj)
            score = board[i_wrapped, j_wrapped] *
                    (1.0 - occupied[i_wrapped, j_wrapped]) *
                    agent.vision_matrix[vision_i, vision_j]
            push!(scores, score)
        end
    end
    move_onehot = differentiable_argmax(scores)
    return move_onehot
end

function move!(board, agent, move_onehot, occupied)
    # reset agent position
    agent.x[1] = zero(agent.x[1])
    agent.y[1] = zero(agent.y[1])
    occupied[agent.i[1], agent.j[1]] = 0.0
    # iterate through all possible moves and assign
    N = size(board, 1)
    max_vision = get_max_vision(agent.vision_matrix)
    move_onehot_index = 1
    for di in (-max_vision):max_vision
        for dj in (-max_vision):max_vision
            i_wrapped, j_wrapped = wrap_index(N, agent.i[1] + di, agent.j[1] + dj)
            agent.x[1] += i_wrapped * move_onehot[move_onehot_index] * agent.alive[1]
            agent.y[1] += j_wrapped * move_onehot[move_onehot_index] * agent.alive[1]
            occupied[i_wrapped, j_wrapped] += move_onehot[move_onehot_index] *
                                              agent.alive[1]
            move_onehot_index += 1
        end
    end
end

function consume!(board, agent, move_onehot)
    # iterate and sum possible wealths
    N = size(board, 1)
    max_vision = get_max_vision(agent.vision_matrix)
    move_onehot_index = 1
    for di in (-max_vision):max_vision
        for dj in (-max_vision):max_vision
            i_wrapped, j_wrapped = wrap_index(N, agent.i[1] + di, agent.j[1] + dj)
            agent.wealth[1] += board[i_wrapped, j_wrapped] *
                               move_onehot[move_onehot_index] * agent.alive[1]
            board[i_wrapped, j_wrapped] *= (1.0 -
                                            move_onehot[move_onehot_index] * agent.alive[1])
            move_onehot_index += 1
        end
    end
    # consume sugar
    agent.wealth[1] -= agent.metabolic_rate[1] * agent.alive[1]
end

function check_death!(agent, occupied, smoothing)
    soft = smoothing(agent.wealth[1])
    hard = ignore_gradient(agent.wealth[1]) >= 0.0
    lives = hard + (soft - ignore_gradient(soft))
    agent.alive[1] = agent.alive[1] * lives
    occupied[agent.i[1], agent.j[1]] = agent.alive[1]
end

function regenerate_sugar!(board, sugar_regeneration_rate, max_sugar_capacities)
    for i in axes(board, 1)
        for j in axes(board, 2)
            board[i, j] = min(board[i, j] + 1.0, max_sugar_capacities[i, j])
            #if rand() < 0.01
            #    board[i, j] += 10.0
            #end
            #else
            #    board[i, j] = 0.0
            #end
        end
    end
end

function sugarscape_abm_step!(
    board, agents, occupied, sugar_regeneration_rate, max_sugar_capacities, smoothing)
    order = sample(1:length(agents), length(agents), replace=false)
    for idx in order
        agent = agents[idx]
        move_onehot = compute_move(board, agent, occupied)
        move!(board, agent, move_onehot, occupied)
        consume!(board, agent, move_onehot)
        check_death!(agent, occupied, smoothing)
    end
    regenerate_sugar!(board, sugar_regeneration_rate, max_sugar_capacities)
end

struct SugarScapeParams{T,Q,R,SM}
    board_initializer::T
    agent_initializer::Q
    board_length::Int64
    n_agents::Int64
    n_timesteps::Int64
    sugar_regeneration_rate::Vector{R}
    gradient_horizon::Int64
    smoothing::SM
end
@functor SugarScapeParams (board_initializer, agent_initializer, sugar_regeneration_rate)

function reset_gradient!(board, agents, occupied, time, gradient_horizon)
    if time % gradient_horizon == 0
        board .= ignore_gradient.(board)
        occupied .= ignore_gradient.(occupied)
        for agent in agents
            agent.x[1] = ignore_gradient(agent.x[1])
            agent.y[1] = ignore_gradient(agent.y[1])
            agent.wealth[1] = ignore_gradient(agent.wealth[1])
            agent.alive[1] = ignore_gradient(agent.alive[1])
        end
    end
end

function abm_run(params::SugarScapeParams)
    diff_type = promote_type(
        typeof(params.sugar_regeneration_rate[1]), get_type(params.board_initializer), get_type(params.agent_initializer))
    board = initialize_board(params.board_initializer, diff_type)
    agents = initialize_agents(params.agent_initializer, params.n_agents, diff_type)
    occupied = zeros(diff_type, params.board_length, params.board_length)
    for agent in agents
        occupied[agent.i[1], agent.j[1]] = 1.0
    end
    board_history = [copy(board)]
    x_history = [[agent.x[1] for agent in agents]]
    y_history = [[agent.y[1] for agent in agents]]
    wealth_history = [[agent.wealth[1] for agent in agents]]
    alive_history = [[agent.alive[1] for agent in agents]]
    occupied_history = [copy(occupied)]
    max_sugar_capacities = copy(board)
    for t in 2:(params.n_timesteps)
        reset_gradient!(board, agents, occupied, t, params.gradient_horizon)
        sugarscape_abm_step!(board, agents, occupied, params.sugar_regeneration_rate,
            max_sugar_capacities, params.smoothing)
        push!(board_history, copy(board))
        push!(x_history, [agent.x[1] for agent in agents])
        push!(y_history, [agent.y[1] for agent in agents])
        push!(wealth_history, [agent.wealth[1] for agent in agents])
        push!(alive_history, [agent.alive[1] for agent in agents])
        push!(occupied_history, copy(occupied))
    end
    return board_history, x_history, y_history, wealth_history, alive_history,
    occupied_history
end

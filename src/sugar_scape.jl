export SugarScapeParams, TwoPeakBoard, RandomAgentInitializer, GeneratedAgentInitializer,
       ConstantAgentInitializer, VonNeumannNeighborhood, MooreNeighborhood, GeneratedBoard

using Base

@kwdef struct SugarSeeker{T, V, M}
    id::Int64
    vision_matrix::V
    metabolic_rate::M
    x::T
    y::T
    alive::T
    wealth::T
end
get_i(agent::SugarSeeker) = Int(round(ignore_gradient(agent.x)))
get_j(agent::SugarSeeker) = Int(round(ignore_gradient(agent.y)))

abstract type BoardInitializer end

struct GeneratedBoard{T} <: BoardInitializer
    N::Int64
    board::Vector{T}
end
get_type(board::GeneratedBoard) = typeof(board.board[1])
@functor GeneratedBoard (board,)

struct TwoPeakBoard{T, Q} <: BoardInitializer
    N::Int
    sugar_peaks::Vector{T}
    max_sugar::Vector{Q}
    distance_function::Function
end
function get_type(board::TwoPeakBoard)
    promote_type(typeof(board.max_sugar[1]), typeof(board.sugar_peaks[1]))
end
@functor TwoPeakBoard (sugar_peaks, max_sugar)

"""
    initialize_board(initializer::TwoPeakBoard, diff_type)

Initialize a Sugarscape environment board with two sugar peaks following exponential decay.

The sugar distribution s(x,y) at each cell (i,j) is computed as:
s(i,j) = Σₖ max_sugar[1] * exp(-distance((i,j), peak_k))

where the sum is over all sugar peaks and distance is computed using the 
specified distance function (typically Manhattan or Euclidean distance).

# Arguments
- `initializer::TwoPeakBoard`: Board configuration with peak locations and maximum sugar values
- `diff_type`: Numeric type for automatic differentiation (e.g., Float64)

# Returns
- `Matrix{diff_type}`: N×N board matrix with sugar reserves at each cell
"""
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
"""
    initialize_board(initializer::GeneratedBoard, diff_type)

Initialize a Sugarscape environment board from pre-generated sugar values.

# Arguments
- `initializer::GeneratedBoard`: Board configuration with pre-computed sugar values
- `diff_type`: Numeric type for automatic differentiation (e.g., Float64)

# Returns
- `Matrix{diff_type}`: N×N board matrix reshaped from the provided sugar values
"""
function initialize_board(initializer::GeneratedBoard, diff_type)
    return reshape(
        copy(convert.(diff_type, initializer.board)), initializer.N, initializer.N)
end

abstract type AgentInitializer end
abstract type Neighborhood end
struct VonNeumannNeighborhood <: Neighborhood end
struct MooreNeighborhood <: Neighborhood end

"""
    is_within_vision(::VonNeumannNeighborhood, i, j, max_vision, vision_radius)

Determine if a cell (i,j) is within vision range using Von Neumann (Manhattan) distance.

This implements the Von Neumann neighborhood where vision is restricted to cells 
whose Manhattan distance from the center is ≤ vision_radius. The condition is:
|i - (max_vision + 1)| + |j - (max_vision + 1)| ≤ vision_radius

# Arguments
- `i, j`: Matrix indices (1-based)
- `max_vision`: Maximum vision radius supported by the matrix
- `vision_radius`: Current agent's vision radius

# Returns
- `Bool`: true if cell is within vision range
"""
function is_within_vision(::VonNeumannNeighborhood, i, j, max_vision, vision_radius)
    abs(i - max_vision - 1) + abs(j - max_vision - 1) <= vision_radius
end

"""
    is_within_vision(::MooreNeighborhood, i, j, max_vision, vision_radius)

Determine if a cell (i,j) is within vision range using Moore (Chebyshev) distance.

This implements the Moore neighborhood where vision extends to all cells within 
a square region. The condition is:
max(|i - (max_vision + 1)|, |j - (max_vision + 1)|) ≤ vision_radius

# Arguments  
- `i, j`: Matrix indices (1-based)
- `max_vision`: Maximum vision radius supported by the matrix
- `vision_radius`: Current agent's vision radius

# Returns
- `Bool`: true if cell is within vision range
"""
function is_within_vision(::MooreNeighborhood, i, j, max_vision, vision_radius)
    max(abs(i - max_vision - 1), abs(j - max_vision - 1)) <= vision_radius
end

"""
    generate_vision_matrices(neighborhood::Neighborhood, max_vision)

Generate vision matrices 𝐌ᵥ for all vision ranges v ∈ {1, 2, ..., max_vision}.

Each vision matrix 𝐌ᵥ is a (2*max_vision + 1) × (2*max_vision + 1) binary matrix where:
- (𝐌ᵥ)ᵢⱼ = 1 if cell (i,j) is visible to an agent with vision radius v
- (𝐌ᵥ)ᵢⱼ = 0 otherwise
- The center position (max_vision+1, max_vision+1) corresponds to the agent's location

The visibility condition depends on the neighborhood type:
- Von Neumann: |i - (max_vision+1)| + |j - (max_vision+1)| ≤ v
- Moore: max(|i - (max_vision+1)|, |j - (max_vision+1)|) ≤ v

These matrices enable matrix-based computation of agent vision as described in the paper.

# Arguments
- `neighborhood::Neighborhood`: Vision pattern (VonNeumannNeighborhood or MooreNeighborhood)
- `max_vision`: Maximum vision radius to support

# Returns
- `Vector{Matrix{Float64}}`: Array of vision matrices, indexed by vision radius
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
"""
    get_max_vision(vision_matrix)

Extract the maximum vision radius from a vision matrix based on its dimensions.

# Arguments
- `vision_matrix`: A (2*max_vision + 1) × (2*max_vision + 1) vision matrix

# Returns
- `Int`: The maximum vision radius max_vision
"""
get_max_vision(vision_matrix) = (size(vision_matrix, 1) - 1) ÷ 2

struct RandomAgentInitializer{T, M, W, U, N, S} <: AgentInitializer
    vision_distribution_probs::Vector{T}
    metabolic_rate_bounds::Vector{M}
    wealth_bounds::Vector{W}
    position_distribution::U
    neighborhood::N
    discrete_sampler::S
    vision_spacing::Int64
end
function RandomAgentInitializer(
        board_length;
        vision_distribution_probs,
        metabolic_rate_bounds,
        wealth_bounds = [8, 10],
        position_distribution = Product(DiscreteUniform.(
            [1, 1], [board_length, board_length])),
        neighborhood = VonNeumannNeighborhood(),
        discrete_sampler = ST(),
        vision_spacing = 1)
    return RandomAgentInitializer(
        vision_distribution_probs,
        metabolic_rate_bounds,
        wealth_bounds,
        position_distribution,
        neighborhood,
        discrete_sampler,
        vision_spacing
    )
end
@functor RandomAgentInitializer (
    vision_distribution_probs, metabolic_rate_bounds, wealth_bounds)
function get_type(initializer::RandomAgentInitializer)
    promote_type(eltype(initializer.vision_distribution_probs),
        eltype(initializer.metabolic_rate_bounds), eltype(initializer.wealth_bounds))
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
function initialize_agents(initializer::RandomAgentInitializer{T, M, W, U, N, S},
        n_agents, diff_type) where {T, M, W, U, N, S}
    agents = SugarSeeker[]
    vision_probs = initializer.vision_distribution_probs
    metabolic_rate_dist = DiffBeta(
        initializer.metabolic_rate_bounds[1], initializer.metabolic_rate_bounds[2])
    wealth_dist = DiffBeta(initializer.wealth_bounds[1], initializer.wealth_bounds[2])

    # Calculate max vision based on vision spacing and number of vision levels
    max_vision = (length(vision_probs) - 1) * initializer.vision_spacing + 1
    all_vision_matrices = generate_vision_matrices(initializer.neighborhood, max_vision)

    # Select vision matrices according to spacing: [1, 1+spacing, 1+2*spacing, ...]
    vision_indices = [1 + i * initializer.vision_spacing
                      for i in 0:(length(vision_probs) - 1)]
    vision_matrices = [all_vision_matrices[idx] for idx in vision_indices]

    for i in 1:n_agents
        vision_onehot_index = rand(DifferentiableOneHotCategorical(
            vision_probs, initializer.discrete_sampler))
        vision_matrix = sum(vision_matrices .* vision_onehot_index)
        metabolic_rate = 2.0 + 2 * rand(metabolic_rate_dist)
        wealth = 6 + 19 * rand(wealth_dist)
        position = rand(initializer.position_distribution)
        agent = SugarSeeker(
            id = i,
            vision_matrix = vision_matrix,
            metabolic_rate = metabolic_rate,
            x = convert(diff_type, position[1]),
            y = convert(diff_type, position[2]),
            alive = one(diff_type),
            wealth = convert(diff_type, wealth))
        push!(agents, agent)
    end
    return agents
end

struct GeneratedAgentInitializer{V, M, W, P, S, N} <: AgentInitializer
    vision_probabilities::Vector{V}
    metabolic_rates::Vector{M}
    wealths::Vector{W}
    positions::Vector{P}
    discrete_sampler::S
    neighborhood::N
    vision_spacing::Int64
end
function GeneratedAgentInitializer(; vision_probabilities, metabolic_rates, wealths,
        positions, discrete_sampler, neighborhood, vision_spacing)
    GeneratedAgentInitializer(vision_probabilities, metabolic_rates, wealths,
        positions, discrete_sampler, neighborhood, vision_spacing)
end
@functor GeneratedAgentInitializer (vision_probabilities, metabolic_rates, wealths)
function get_type(initializer::GeneratedAgentInitializer)
    promote_type(eltype(initializer.metabolic_rates), eltype(initializer.wealths),
        eltype(initializer.vision_probabilities))
end
"""
    initialize_agents(initializer::GeneratedAgentInitializer, n_agents, diff_type, vision_spacing)

Initialize Sugarscape agents from pre-generated parameter distributions.

This function creates agents with vision matrices sampled from a categorical distribution
using the straight-through (ST) estimator. During the primal pass, each agent receives
a discrete vision matrix, while during the tangent pass, agents use the weighted average:
𝐌̃ = Σᵥ pᵥ 𝐌ᵥ where pᵥ are the vision probabilities.

# Arguments
- `initializer::GeneratedAgentInitializer`: Pre-computed agent parameters
- `n_agents`: Number of agents to create
- `diff_type`: Numeric type for automatic differentiation
- `vision_spacing`: Spacing parameter for vision levels (unused in this method)

# Returns
- `Vector{SugarSeeker}`: Array of initialized agents with sampled parameters
"""
function initialize_agents(
        initializer::GeneratedAgentInitializer, n_agents, diff_type, vision_spacing)
    max_vision = length(initializer.vision_probabilities)
    vision_matrices = generate_vision_matrices(initializer.neighborhood, max_vision)
    vision_probs = initializer.vision_probabilities ./ sum(initializer.vision_probabilities)
    agents = SugarSeeker[]
    for i in 1:n_agents
        vision_onehot_index = rand(DifferentiableOneHotCategorical(
            vision_probs, initializer.discrete_sampler))
        vision_matrix = sum(vision_matrices .* vision_onehot_index)
        position = initializer.positions[i]
        push!(agents,
            SugarSeeker(
                id = i,
                vision_matrix = vision_matrix,
                metabolic_rate = initializer.metabolic_rates[i],
                x = convert(diff_type, position[1]),
                y = convert(diff_type, position[2]),
                alive = one(diff_type),
                wealth = convert(diff_type, initializer.wealths[i])))
    end
    return agents
end

"""
    compute_move(board, agent, occupied)

Compute an agent's optimal movement decision using differentiable cell selection.

This function implements the core movement logic where agents choose their destination
by maximizing the score function 𝐙ᵢ,ₜ = 𝐒ᵢ,ₜ ⊙ 𝐎ᵢ,ₜ ⊙ 𝐌ᵥᵢ, where:
- 𝐒ᵢ,ₜ: local sugar matrix around agent i
- 𝐎ᵢ,ₜ: local occupancy matrix (1 - occupied)  
- 𝐌ᵥᵢ: agent i's vision matrix

The argmax operation uses surrogate gradients (straight-through estimator) to enable
differentiation through the discrete cell selection.

# Arguments
- `board`: Global sugar distribution matrix
- `agent`: SugarSeeker agent with vision matrix
- `occupied`: Global occupancy matrix (1 = occupied, 0 = free)

# Returns
- `Vector`: One-hot vector indicating the selected movement destination
"""
function compute_move(board, agent, occupied)
    scores = eltype(board)[]
    max_vision = get_max_vision(agent.vision_matrix)
    N = size(board, 1)
    for (vision_i, di) in enumerate((-max_vision):max_vision)
        for (vision_j, dj) in enumerate((-max_vision):max_vision)
            i_wrapped, j_wrapped = wrap_index(N, get_i(agent) + di, get_j(agent) + dj)
            score = board[i_wrapped, j_wrapped] *
                    (1.0 - occupied[i_wrapped, j_wrapped]) *
                    agent.vision_matrix[vision_i, vision_j]
            push!(scores, score)
        end
    end
    move_onehot = differentiable_argmax(scores)
    return move_onehot
end

"""
    compute_new_position(board, agent, move_onehot, occupied)

Update agent position based on movement decision using shift matrices.

This function translates the matrix-based movement selection back to spatial coordinates
using the shift matrices Δₓ and Δᵧ. The position updates are:
- xᵢ(t+1) = xᵢ(t) + ⟨𝐙*ᵢ,ₜ, Δₓ⟩  
- yᵢ(t+1) = yᵢ(t) + ⟨𝐙*ᵢ,ₜ, Δᵧ⟩

where ⟨·,·⟩ denotes the Frobenius inner product and 𝐙*ᵢ,ₜ is the one-hot movement matrix.

# Arguments
- `board`: Global sugar distribution matrix
- `agent`: SugarSeeker agent making the move
- `move_onehot`: One-hot vector indicating selected destination
- `occupied`: Global occupancy matrix (modified in-place)

# Returns
- `Tuple{T, T}`: New (x, y) coordinates for the agent
"""
function compute_new_position(board, agent, move_onehot, occupied)
    # reset agent position
    new_x = zero(agent.x)
    new_y = zero(agent.y)
    i_wrapped, j_wrapped = wrap_index(size(board, 1), get_i(agent), get_j(agent))
    occupied[i_wrapped, j_wrapped] = 0.0
    # iterate through all possible moves and assign
    N = size(board, 1)
    max_vision = get_max_vision(agent.vision_matrix)
    move_onehot_index = 1
    for di in (-max_vision):max_vision
        for dj in (-max_vision):max_vision
            i_wrapped, j_wrapped = wrap_index(N, get_i(agent) + di, get_j(agent) + dj)
            new_x += i_wrapped * move_onehot[move_onehot_index] * agent.alive
            new_y += j_wrapped * move_onehot[move_onehot_index] * agent.alive
            occupied[i_wrapped, j_wrapped] += move_onehot[move_onehot_index] * agent.alive
            move_onehot_index += 1
        end
    end
    return new_x, new_y
end

"""
    compute_new_wealth(board, agent, move_onehot)

Update agent wealth based on sugar harvesting and metabolic consumption.

This function implements the wealth update equation:
hᵢ(t+1) = hᵢ(t) + [⟨𝐙*ᵢ,ₜ, 𝐒ᵢ,ₜ⟩ - mᵢ] · aᵢ(t)

where hᵢ(t) is wealth, 𝐙*ᵢ,ₜ is the movement matrix, 𝐒ᵢ,ₜ is the local sugar matrix,
mᵢ is metabolic rate, and aᵢ(t) is the alive indicator.

# Arguments
- `board`: Global sugar distribution matrix (modified in-place)
- `agent`: SugarSeeker agent harvesting sugar
- `move_onehot`: One-hot vector indicating selected destination

# Returns
- `T`: New wealth value for the agent
"""
function compute_new_wealth(board, agent, move_onehot)
    # iterate and sum possible wealths
    N = size(board, 1)
    max_vision = get_max_vision(agent.vision_matrix)
    move_onehot_index = 1
    new_wealth = agent.wealth
    for di in (-max_vision):max_vision
        for dj in (-max_vision):max_vision
            i_wrapped, j_wrapped = wrap_index(N, get_i(agent) + di, get_j(agent) + dj)
            new_wealth += board[i_wrapped, j_wrapped] *
                          move_onehot[move_onehot_index] * agent.alive
            board[i_wrapped, j_wrapped] *= (1.0 -
                                            move_onehot[move_onehot_index] * agent.alive)
            move_onehot_index += 1
        end
    end
    # consume sugar
    new_wealth -= agent.metabolic_rate * agent.alive
    return new_wealth
end

"""
    compute_alive(agent, occupied, smoothing)

Determine agent survival using differentiable binary decisions.

This function handles the non-differentiable survival predicate (wealth ≤ 0) by using
surrogate gradients. During the primal pass, agents die if wealth ≤ 0. During the 
tangent pass, the sigmoid approximation smooths the binary life-or-death decision.

# Arguments
- `agent`: SugarSeeker agent to check for survival
- `occupied`: Global occupancy matrix (unused)
- `smoothing`: Sigmoid-like function for smooth survival transitions

# Returns
- `T`: New alive indicator (0.0 = dead, 1.0 = alive)
"""
function compute_alive(agent, occupied, smoothing)
    #return agent.alive
    wealth = agent.wealth[1]
    soft = smoothing(wealth)
    hard = ignore_gradient(wealth) >= 0.0
    lives = hard + (soft - ignore_gradient(soft))
    new_alive = agent.alive * lives
    return new_alive
end

function regenerate_sugar!(board, sugar_regeneration_rate, max_sugar_capacities)
    for i in axes(board, 1)
        for j in axes(board, 2)
            board[i, j] = min(board[i, j] + 1.0, max_sugar_capacities[i, j])
        end
    end
end

function sugarscape_abm_step!(
        board, agents, occupied, sugar_regeneration_rate, max_sugar_capacities, smoothing)
    order = sample(1:length(agents), length(agents), replace = false)
    for idx in order
        agent = agents[idx]
        move_onehot = compute_move(board, agent, occupied)
        new_x, new_y = compute_new_position(board, agent, move_onehot, occupied)
        new_wealth = compute_new_wealth(board, agent, move_onehot)
        new_alive = compute_alive(agent, occupied, smoothing)
        agents[idx] = SugarSeeker(
            id = agent.id,
            vision_matrix = agent.vision_matrix,
            metabolic_rate = agent.metabolic_rate,
            x = new_x,
            y = new_y,
            alive = new_alive,
            wealth = new_wealth
        )
    end
    regenerate_sugar!(board, sugar_regeneration_rate, max_sugar_capacities)
end

struct SugarScapeParams{T, Q, R, SM}
    board_initializer::T
    agent_initializer::Q
    board_length::Int64
    n_agents::Int64
    n_timesteps::Int64
    sugar_regeneration_rate::Vector{R}
    gradient_horizon::Int64
    smoothing::SM
end
@functor SugarScapeParams (agent_initializer,)

function reset_gradient!(board, agents, occupied, time, gradient_horizon)
    if time % gradient_horizon == 0
        board .= ignore_gradient.(board)
        occupied .= ignore_gradient.(occupied)
        for idx in 1:length(agents)
            agents[idx] = SugarSeeker(
                id = agents[idx].id,
                vision_matrix = agents[idx].vision_matrix,
                metabolic_rate = agents[idx].metabolic_rate,
                x = ignore_gradient(agents[idx].x),
                y = ignore_gradient(agents[idx].y),
                alive = ignore_gradient(agents[idx].alive),
                wealth = ignore_gradient(agents[idx].wealth)
            )
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
        occupied[get_i(agent), get_j(agent)] = 1.0
    end
    board_history = [copy(board)]
    x_history = [[agent.x for agent in agents]]
    y_history = [[agent.y for agent in agents]]
    wealth_history = [[agent.wealth for agent in agents]]
    alive_history = [[agent.alive for agent in agents]]
    occupied_history = [copy(occupied)]
    max_sugar_capacities = copy(board)
    for t in 2:(params.n_timesteps)
        reset_gradient!(board, agents, occupied, t, params.gradient_horizon)
        sugarscape_abm_step!(board, agents, occupied, params.sugar_regeneration_rate,
            max_sugar_capacities, params.smoothing)
        push!(board_history, copy(board))
        push!(x_history, [agent.x for agent in agents])
        push!(y_history, [agent.y for agent in agents])
        push!(wealth_history, [agent.wealth for agent in agents])
        push!(alive_history, [agent.alive for agent in agents])
        push!(occupied_history, copy(occupied))
    end
    return board_history, x_history, y_history, wealth_history, alive_history,
    occupied_history
end
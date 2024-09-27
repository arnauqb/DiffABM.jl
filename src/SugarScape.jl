export SugarScapeParams, TwoPeakBoard, RandomAgentInitializer, ArgmaxMovingRule, VonNeumannNeighborhood, MooreNeighborhood

using Base

function is_out_of_bounds(N, i, j)
	return i < 1 || i > N || j < 1 || j > N
end

function wrap_index(N, i, j)
	# given board length N, and index i, j
	# return the wrapped index to make the board toroidal
	return mod(i - 1, N) + 1, mod(j - 1, N) + 1
end

struct SugarSeeker{T, V, M}
	id::Int64
	age::Vector{T}
	max_age::Float64
	vision::V
	metabolic_rate::M
	x::Vector{T}
	y::Vector{T}
	alive::Vector{T}
	wealth::Vector{T}
end

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
get_type(board::TwoPeakBoard) = promote_type(typeof(board.max_sugar[1]), typeof(board.sugar_peaks[1]))
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
                dist = initializer.distance_function((i, j), (sugar_peak_xs[k], sugar_peak_ys[k])) + 1e-3
				sugar = max_sugar[1] * exp(-dist)
				board[i, j] += sugar
			end
		end
	end
	return board
end
function initialize_board(initializer::GeneratedBoard, diff_type)
	return reshape(initializer.board, initializer.N, initializer.N)
end

abstract type AgentInitializer end

struct RandomAgentInitializer{T, Q, R, S, U} <: AgentInitializer
	vision_distribution::T
	metabolic_rate_distribution::R
	max_age_distribution::Q
	wealth_distribution::S
	position_distribution::U
end
function RandomAgentInitializer(
	board_length;
	vision_distribution = DiscreteUniform(1, 5),
	metabolic_rate_distribution = Uniform(1, 4),
	max_age_distribution = Uniform(60, 100),
	wealth_distribution = Uniform(10, 20),
	position_distribution = Product(DiscreteUniform.([1, 1], [board_length, board_length])),
)
	return RandomAgentInitializer(
		vision_distribution,
		metabolic_rate_distribution,
		max_age_distribution,
		wealth_distribution,
		position_distribution,
	)
end
@functor RandomAgentInitializer (vision_distribution, metabolic_rate_distribution, max_age_distribution, wealth_distribution, position_distribution)

"""
	initialize_agents(initializer::RandomAgentInitializer{T, Q, R, S, U}, n_agents, diff_type) where {T, Q, R, S, U}

Initialize a list of agents for the Sugarscape simulation.

# Arguments
- `initializer::RandomAgentInitializer{T, Q, R, S, U}`: An instance of `RandomAgentInitializer` containing the distributions for agent attributes.
- `n_agents`: The number of agents to initialize.
- `diff_type`: The type used for differentiation (e.g., `Float64`).

# Returns
- A list of `SugarSeeker` agents initialized with random attributes based on the provided distributions.
"""
function initialize_agents(initializer::RandomAgentInitializer{T, Q, R, S, U}, n_agents, diff_type) where {T, Q, R, S, U}
	agents = SugarSeeker[]
	for i in 1:n_agents
		vision = rand(initializer.vision_distribution)
		metabolic_rate = rand(initializer.metabolic_rate_distribution)
		max_age = rand(initializer.max_age_distribution)
		wealth = convert(diff_type, rand(initializer.wealth_distribution))
		position = convert.(diff_type, rand(initializer.position_distribution))
		agent = SugarSeeker(
			i,
			[zero(diff_type)],
			max_age,
			vision,
			metabolic_rate,
			[position[1]],
			[position[2]],
			[one(diff_type)],
			[wealth])
		push!(agents, agent)
	end
	return agents
end

abstract type MovingRule end

abstract type Neighborhood end

make_vision_matrix(max_vision) = [[i >= j ? 1 : 0 for j in 1:max_vision] for i in 1:max_vision]
struct VonNeumannNeighborhood <: Neighborhood
	board_length::Int64
	max_vision::Int64
	vision_matrix::Vector{Vector{Float64}}
end
function VonNeumannNeighborhood(board_length, vision)
	vision_matrix = make_vision_matrix(vision)
	return VonNeumannNeighborhood(board_length, vision, vision_matrix)
end
function iterate(vnn::VonNeumannNeighborhood, i, j, vision)
	ret = typeof((i, j))[]
	for direction in 1:4
		if direction == 1
			for k in 1:vision
				if j + k > vnn.board_length
					continue
				end
				push!(ret, (i, j + k))
			end
		elseif direction == 2
			for k in 1:vision
				if j - k < 1
					continue
				end
				push!(ret, (i, j - k))
			end
		elseif direction == 3
			for k in 1:vision
				if i + k > vnn.board_length
					continue
				end
				push!(ret, (i + k, j))
			end
		elseif direction == 4
			for k in 1:vision
				if i - k < 1
					continue
				end
				push!(ret, (i - k, j))
			end
		end
	end
	return ret
end

struct MooreNeighborhood <: Neighborhood
	board_length::Int64
	max_vision::Int64
	vision_matrix::Vector{Vector{Float64}}
end
function MooreNeighborhood(board_length, max_vision)
	vision_matrix = [[i >= j ? 1 : 0 for j in 1:max_vision] for i in 1:max_vision]
	return MooreNeighborhood(board_length, max_vision, vision_matrix)
end
function iterate(mnn::MooreNeighborhood, i, j, vision)
	# iterate over all cells within vision distance
	ret = typeof((i, j))[]
	for di in -vision:vision
		for dj in -vision:vision
			if di == 0 && dj == 0
				continue  # Skip the center cell
			end
			row, col = i + di, j + dj
			if 1 <= row <= mnn.board_length && 1 <= col <= mnn.board_length
				push!(ret, (row, col))
			end
		end
	end
	return ret
end


"""
ArgmaxMovingRule: move to the cell with the maximum sugar
"""
struct ArgmaxMovingRule{T} <: MovingRule
	neighborhood::T
end

function compute_move(board, agent, rule::ArgmaxMovingRule, occupied)
	scores = eltype(board)[]
	#for (position, vision_mask) in iterate(rule.neighborhood, agent.x[1], agent.y[1], agent.vision[1])
	for position in iterate(rule.neighborhood, agent.x[1], agent.y[1], agent.vision[1])
		#push!(scores, board[Int64(position[1]), Int64(position[2])] * vision_mask)
        sugar = board[Int64(round(position[1])), Int64(round(position[2]))]
        score = sugar #* (1.0 - occupied[Int64(round(position[1])), Int64(round(position[2]))])
		push!(scores, score)
    end
	move_onehot = differentiable_argmax(scores)
	return move_onehot
end

function move!(board, agent, move_onehot, rule::ArgmaxMovingRule, occupied)
	new_x = zero(eltype(board))
	new_y = zero(eltype(board))
	#for (i, (position, _)) in enumerate(iterate(rule.neighborhood, agent.x[1], agent.y[1], agent.vision[1]))
    #occupied[Int64(round(agent.x[1])), Int64(round(agent.y[1]))] -= agent.alive[1]
	for (i, position) in enumerate(iterate(rule.neighborhood, agent.x[1], agent.y[1], agent.vision[1]))
		new_x += position[1] * move_onehot[i]
		new_y += position[2] * move_onehot[i]
        #occupied[Int64(round(position[1])), Int64(round(position[2]))] += move_onehot[i] * agent.alive[1]
	end
	agent.x[1] = agent.x[1] + agent.alive[1] * (new_x - agent.x[1])
	agent.y[1] = agent.y[1] + agent.alive[1] * (new_y - agent.y[1])
end

function consume!(board, agent, neighbourhood, move_onehot)
	new_wealth = agent.wealth[1]
	#for (i, (position, _)) in enumerate(iterate(neighbourhood, agent.x[1], agent.y[1], agent.vision[1]))
	for (i, position) in enumerate(iterate(neighbourhood, agent.x[1], agent.y[1], agent.vision[1]))
		new_wealth += board[Int64(round(position[1])), Int64(round(position[2]))] * move_onehot[i] * agent.alive[1]
		board[Int64(round(position[1])), Int64(round(position[2]))] = board[Int64(round(position[1])), Int64(round(position[2]))] * (1.0 - move_onehot[i] * agent.alive[1])
	end
	agent.wealth[1] = max(zero(eltype(board)), agent.wealth[1] + agent.alive[1] * (new_wealth - agent.wealth[1]) - agent.metabolic_rate[1])
end

function check_death!(agent, occupied)
	no_wealth = differentiable_is_less(SigmoidSmoothing(1.0), agent.wealth[1], 0.0)
	#old = differentiable_is_greater(SigmoidSmoothing(1.0), agent.age[1], agent.max_age)
	#is_dead = differentiable_or(no_wealth, old)
	agent.alive[1] = agent.alive[1] * differentiable_not(no_wealth)
    #occupied[Int64(round(agent.x[1])), Int64(round(agent.y[1]))] = agent.alive[1]
end

function age!(agent)
	agent.age[1] = agent.age[1] + agent.alive[1]
end

function regenerate_sugar!(board, sugar_regeneration_rate, max_sugar_capacities)
	for i in axes(board, 1)
		for j in axes(board, 2)
			#board[i, j] = min(board[i, j] + sugar_regeneration_rate[1], max_sugar_capacities[i, j])
			board[i, j] += sugar_regeneration_rate[1] * max_sugar_capacities[i, j]
		end
	end
end

function abm_step!(board, agents, occupied, moving_rule::ArgmaxMovingRule, sugar_regeneration_rate, max_sugar_capacities)
	order = sample(1:length(agents), length(agents), replace = false)
	for idx in order
		agent = agents[idx]
		move_onehot = compute_move(board, agent, moving_rule, occupied)
		consume!(board, agent, moving_rule.neighborhood, move_onehot)
		#age!(agent)
		#check_death!(agent, occupied)
		move!(board, agent, move_onehot, moving_rule, occupied)
	end
	regenerate_sugar!(board, sugar_regeneration_rate, max_sugar_capacities)
end

struct SugarScapeParams{T, Q, S, R}
	board_initializer::T
	agent_initializer::Q
	moving_rule::S
	board_length::Int64
	n_agents::Int64
	n_timesteps::Int64
	sugar_regeneration_rate::Vector{R}
end
@functor SugarScapeParams (board_initializer, agent_initializer, sugar_regeneration_rate)

function abm_run(params::SugarScapeParams)
	diff_type = get_type(params.board_initializer)
	board = initialize_board(params.board_initializer, diff_type)
	agents = initialize_agents(params.agent_initializer, params.n_agents, diff_type)
    occupied = zeros(diff_type, params.board_length, params.board_length)
    for agent in agents
        occupied[Int64(round(agent.x[1])), Int64(round(agent.y[1]))] = 1.0
    end
	board_history = [copy(board)]
	x_history = [[agent.x[1] for agent in agents]]
	y_history = [[agent.y[1] for agent in agents]]
	alive_history = [[agent.alive[1] for agent in agents]]
	max_sugar_capacities = copy(board)
	for t in 1:params.n_timesteps
		abm_step!(board, agents, occupied, params.moving_rule, params.sugar_regeneration_rate, max_sugar_capacities)
		push!(board_history, copy(board))
		push!(x_history, [agent.x[1] for agent in agents])
		push!(y_history, [agent.y[1] for agent in agents])
		push!(alive_history, [agent.alive[1] for agent in agents])
	end
	return board_history, x_history, y_history, alive_history
end

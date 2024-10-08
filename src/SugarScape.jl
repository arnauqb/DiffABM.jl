export SugarScapeParams, TwoPeakBoard, RandomAgentInitializer, GeneratedAgentInitializer, ConstantAgentInitializer,
	ArgmaxMovingRule, VonNeumannNeighborhood, MooreNeighborhood, GeneratedBoard

using Base

function is_out_of_bounds(N, i, j)
	return i < 1 || i > N || j < 1 || j > N
end

Base.isinteger(x::Real, tol::Float64) = abs(round(x) - x) < tol

function Base.getindex(board::AbstractArray, i::T,
	j::T) where {T <: Union{Float64, ForwardDiff.Dual, StochasticAD.StochasticTriple}}
	i = StochasticAD.value(i)
	j = StochasticAD.value(j)
	#@assert isinteger(i, 1e-6)
	#@assert isinteger(j, 1e-6)
	return board[Int64(round(i)), Int64(round(j))]
end
function Base.setindex!(board::AbstractArray, v, i::T,
	j::T) where {T <: Union{Float64, ForwardDiff.Dual, StochasticAD.StochasticTriple}}
	i = StochasticAD.value(i)
	j = StochasticAD.value(j)
	#@assert isinteger(i, 1e-6)
	#@assert isinteger(j, 1e-6)
	return board[Int64(round(i)), Int64(round(j))] = v
end

function wrap_index(N, i, j)
	# given board length N, and index i, j
	# return the wrapped index to make the board toroidal
	return mod(i - 1, N) + 1, mod(j - 1, N) + 1
end

function check_delta_s(x)
	if typeof(x) <: StochasticAD.StochasticTriple
		if isnan(StochasticAD.get_Δs(x, StochasticAD.PrunedFIsBackend()).Δ)
			return true
		end
	end
	return false
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
	return reshape(copy(convert.(diff_type, initializer.board)), initializer.N, initializer.N)
end

abstract type AgentInitializer end

struct RandomAgentInitializer{T, Q, R, S, U} <: AgentInitializer
	vision_distribution_probs::Vector{T}
	metabolic_rate_probs::Vector{R}
	max_age_distribution::Q
	wealth_distribution::S
	position_distribution::U
end
function RandomAgentInitializer(
	board_length;
	vision_distribution_probs,
	metabolic_rate_probs,
	max_age_distribution = Uniform(60, 100),
	wealth_distribution = Uniform(8, 10),
	position_distribution = Product(DiscreteUniform.(
		[1, 1], [board_length, board_length])),
)
	return RandomAgentInitializer(
		vision_distribution_probs,
		metabolic_rate_probs,
		max_age_distribution,
		wealth_distribution,
		position_distribution,
	)
end
@functor RandomAgentInitializer (vision_distribution_probs, metabolic_rate_probs)
get_type(initializer::RandomAgentInitializer) = typeof(initializer.vision_distribution_probs[1])

struct GeneratedAgentInitializer{T, Q, R, S, U} <: AgentInitializer
	visions::Vector{T}
	metabolic_rates::Vector{Q}
	max_ages::Vector{R}
	wealths::Vector{S}
	positions::Vector{U}
end

struct ConstantAgentInitializer{T, Q, R, S, U} <: AgentInitializer
	vision::Vector{T}
	metabolic_rate::Vector{Q}
	max_age::Vector{R}
	wealth::Vector{S}
	positions::Vector{U}
end
@functor ConstantAgentInitializer (vision, metabolic_rate, max_age, wealth)
get_type(initializer::ConstantAgentInitializer) = typeof(initializer.vision[1])

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
function initialize_agents(initializer::RandomAgentInitializer{T, Q, R, S, U},
	n_agents, diff_type) where {T, Q, R, S, U}
	agents = SugarSeeker[]
    # correct numerical error
    #vision_bounds = initializer.vision_distribution_probs
    vision_probs = initializer.vision_distribution_probs
    vision_probs = vision_probs ./ sum(vision_probs)
    vision_probs = reshape(vision_probs, :, 1)
    metabolic_rate_probs = initializer.metabolic_rate_probs
    metabolic_rate_probs = metabolic_rate_probs ./ sum(metabolic_rate_probs)
    metabolic_rate_probs = reshape(metabolic_rate_probs, :, 1)
	for i in 1:n_agents
		#vision = vision_bounds[1] + rand() * (vision_bounds[2] - vision_bounds[1])
        #vision = 3 * sample_categorical(SAD(), vision_probs)[1]
        vision = 2 * sample_categorical(SAD(), vision_probs)[1] - 1.0
		metabolic_rate = 4 *  sample_categorical(SAD(), metabolic_rate_probs)[1] - 3
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

function initialize_agents(initializer::GeneratedAgentInitializer, n_agents, diff_type)
	agents = SugarSeeker[]
	for i in 1:n_agents
		vision = initializer.visions[i]
		metabolic_rate = initializer.metabolic_rates[i]
		max_age = initializer.max_ages[i]
		wealth = convert(diff_type, initializer.wealths[i])
		position = convert.(diff_type, initializer.positions[i])
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

function initialize_agents(initializer::ConstantAgentInitializer, n_agents, diff_type)
	agents = SugarSeeker[]
	for i in 1:n_agents
		vision = initializer.vision[1]
		metabolic_rate = initializer.metabolic_rate[1]
		max_age = initializer.max_age[1]
		wealth = convert(diff_type, initializer.wealth[1])
		position = convert.(diff_type, initializer.positions[i])
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

function make_vision_matrix(max_vision)
	[[i >= j ? 1 : 0 for j in 1:max_vision] for i in 1:max_vision]
end
struct VonNeumannNeighborhood <: Neighborhood
	board_length::Int64
	max_vision::Int64
	vision_matrix::Vector{Vector{Float64}}
end
function VonNeumannNeighborhood(board_length, vision)
	vision_matrix = make_vision_matrix(vision)
	return VonNeumannNeighborhood(board_length, vision, vision_matrix)
end
function iterate(vnn::VonNeumannNeighborhood, i, j, vision::T) where T
	ret = Tuple{Tuple{Float64, Float64}, T}[]
    max_vision = vnn.max_vision
    i = StochasticAD.value(i)
    j = StochasticAD.value(j)
	for di in (-max_vision):max_vision
		for dj in (-max_vision):max_vision
            if di == 0 && dj == 0
                continue
            end
            row, col = wrap_index(vnn.board_length, i + di, j + dj)
            distance = sqrt(di^2 + dj^2)
            has_vision_hard = ignore_gradient(distance) <= ignore_gradient(vision)
            #has_vision_soft = GaussianSmoothing(0.1)(vision - distance)
            has_vision_soft = PiecewiseSmoothing([-0.5, 0.5])(vision - distance)
            has_vision = has_vision_hard + (has_vision_soft - ignore_gradient(has_vision_soft))
            push!(ret, ((row, col), has_vision))
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
function iterate(mnn::MooreNeighborhood, i, j, vision::T) where T
	# iterate over all cells within vision distance
	i = StochasticAD.value(i)
	j = StochasticAD.value(j)
	#vision = StochasticAD.value(vision)
	max_vision = mnn.max_vision
	ret = Tuple{Tuple{Float64, Float64}, T}[]
	for di in (-max_vision):max_vision
		for dj in (-max_vision):max_vision
			if di == 0 && dj == 0
				continue  # Skip the center cell
			end
			row, col = wrap_index(mnn.board_length, i + di, j + dj)
			distance = max(abs(di), abs(dj)) #sqrt(di^2 + dj^2)
			has_vision_hard = ignore_gradient(distance) <= ignore_gradient(vision)
			has_vision_soft = GaussianSmoothing(0.1)(vision - distance)
			has_vision = has_vision_hard + (has_vision_soft - ignore_gradient(has_vision_soft))
			push!(ret, ((row, col), has_vision))
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
    vision_iterator = iterate(rule.neighborhood, agent.x[1], agent.y[1], agent.vision[1])
	for (position, has_vision) in vision_iterator
		sugar = board[position[1], position[2]]
		score = sugar * (1.0 - occupied[position[1], position[2]]) * has_vision
		push!(scores, score)
	end
	move_onehot = differentiable_argmax(scores)
	return move_onehot, vision_iterator
end

function move!(board, agent, move_onehot, vision_iterator, occupied)
	new_x = zero(eltype(board))
	new_y = zero(eltype(board))
	occupied[agent.x[1], agent.y[1]] -= agent.alive[1] * sum(move_onehot)
	for (i, (position, _)) in enumerate(vision_iterator)
		new_x += position[1] * move_onehot[i]
		new_y += position[2] * move_onehot[i]
		occupied[position[1], position[2]] += move_onehot[i] * agent.alive[1]
	end
	agent.x[1] = agent.x[1] + agent.alive[1] * (new_x - agent.x[1])
	agent.y[1] = agent.y[1] + agent.alive[1] * (new_y - agent.y[1])
end

function consume!(board, agent, vision_iterator, move_onehot)
	new_wealth = agent.wealth[1]
	for (i, (position, _)) in enumerate(vision_iterator)
		new_wealth += board[position[1], position[2]] * move_onehot[i] * agent.alive[1]
		new_sugar = board[position[1], position[2]] *
					(1.0 - move_onehot[i] * agent.alive[1])
		board[position[1], position[2]] = new_sugar
	end
	#agent.wealth[1] = max(0.0, agent.alive[1] * (agent.wealth[1] + (new_wealth - agent.wealth[1]) -
	agent.wealth[1] = agent.alive[1] * (new_wealth - agent.metabolic_rate[1])
end

function check_death!(agent, occupied, average_wealth)
	# line that is y=1 at x=0.0 and y=0 at x=10.0
	#soft = max(0.0, 5.0 - agent.wealth[1]) / 5.0
	#soft = max(0.0, 1.0 - agent.wealth[1])
	#if agent.wealth[1] > 10.0
	#soft = 0.0
	#else
	#soft = 1.0 - GaussianSmoothing(0.1)(agent.wealth[1] / average_wealth)
    soft = 1.0 - PiecewiseSmoothing([-0.5, 0.5])(agent.wealth[1] / average_wealth)
	#soft = 1.0 - SigmoidSmoothing(average_wealth)(agent.wealth[1])
	#soft = clamp(soft, 1e-8, 1.0)
	#println(soft)
	#end
	hard = ignore_gradient(agent.wealth[1]) <= 0.0
	dies = hard + (soft - ignore_gradient(soft))
	agent.alive[1] = agent.alive[1] * (1.0 - dies)
	occupied[agent.x[1], agent.y[1]] = agent.alive[1]
end

function age!(agent)
	agent.age[1] = agent.age[1] + agent.alive[1]
end

function regenerate_sugar!(board, sugar_regeneration_rate, max_sugar_capacities)
	for i in axes(board, 1)
		for j in axes(board, 2)
			#board[i, j] += 0.1 * max_sugar_capacities[i, j] #min(board[i, j] + 1.0, max_sugar_capacities[i, j])
            #if rand() < 0.01
			    #board#[i, j] += 10.0 #max_sugar_capacities[i, j] #min(board[i, j] + 1.0, max_sugar_capacities[i, j])
            #end
            board[i, j] = min(board[i, j] + 1.0, max_sugar_capacities[i, j])
		end
	end
end

function abm_step!(board, agents, occupied, moving_rule::ArgmaxMovingRule,
	sugar_regeneration_rate, max_sugar_capacities)
	order = sample(1:length(agents), length(agents), replace = false)
	nonzero_wealths = [ignore_gradient(agent.wealth[1]) for agent in agents if ignore_gradient(agent.wealth[1]) > 0.0]
	if length(nonzero_wealths) == 0
        average_wealth = 0.0
	else
		average_wealth = mean(nonzero_wealths)
	end
	for idx in order
		agent = agents[idx]
		move_onehot, vision_iterator = compute_move(board, agent, moving_rule, occupied)
		consume!(board, agent, vision_iterator, move_onehot)
		age!(agent)
		check_death!(agent, occupied, average_wealth)
		move!(board, agent, move_onehot, vision_iterator, occupied)
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
	gradient_horizon::Int64
end
@functor SugarScapeParams (board_initializer, agent_initializer, sugar_regeneration_rate)

function adjust_gradient(x::ForwardDiff.Dual, grad_div::ForwardDiff.Dual)
	x_partials = x.partials
	grad_div_partials = grad_div.partials
	new_partial_values = Tuple(x_partials[i] /
							   (grad_div_partials[i] == 0 ? 1.0 : grad_div_partials[i])
							   for i in axes(x_partials, 1))
	new_partials = typeof(x_partials)(new_partial_values)
	return typeof(x)(x.value, new_partials)
end
adjust_gradient(x, grad_div) = x

function compute_discounted_gradient(history, gamma)
	return [history[i] / gamma^(gradient_horizon - i) for i in 1:gradient_horizon]
end

function reset_rolling_gradient!(
	board, agents, occupied, board_history, x_history, y_history,
	wealth_history, alive_history, occupied_history, time, gradient_horizon)
	"""
	We need to derive each gradient by the one gradient_horizon steps before.
	"""
	if time <= gradient_horizon
		return
	end
	#println("board: ", board[1, 1])
	#println("board_history: ", board_history[time - gradient_horizon][1, 1])
	board .= adjust_gradient.(board, board_history[time-gradient_horizon])
	#println("adjusted board: ", board[1, 1])
	#println("occupied: ", occupied[1, 1])
	#println("occupied_history: ", occupied_history[time - gradient_horizon][1, 1])
	occupied .= adjust_gradient.(occupied, occupied_history[time-gradient_horizon])
	#println("adjusted occupied: ", occupied[1, 1])
	#println(sum(board))
	for (i, agent) in enumerate(agents)
		#println("**")
		#println("agent $i")
		agent.x[1] = adjust_gradient(agent.x[1], x_history[time-gradient_horizon][i])
		#println("agent $i x: ", agent.x[1])
		agent.y[1] = adjust_gradient(agent.y[1], y_history[time-gradient_horizon][i])
		#println("agent $i y: ", agent.y[1])
		agent.wealth[1] = adjust_gradient(
			agent.wealth[1], wealth_history[time-gradient_horizon][i])
		#println("agent $i wealth: ", agent.wealth[1])
		agent.alive[1] = adjust_gradient(
			agent.alive[1], alive_history[time-gradient_horizon][i])
		#println("agent $i alive: ", agent.alive[1])
	end
	#println("adjusted agent wealth: ", agents[1].wealth[1])
	#println("-------")
end

function reset_gradient!(board, agents, occupied, board_history, x_history, y_history,
	wealth_history, alive_history, occupied_history, time, gradient_horizon)
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
		occupied[agent.x[1], agent.y[1]] = 1.0
	end
	board_history = [copy(board)]
	x_history = [[agent.x[1] for agent in agents]]
	y_history = [[agent.y[1] for agent in agents]]
	wealth_history = [[agent.wealth[1] for agent in agents]]
	alive_history = [[agent.alive[1] for agent in agents]]
	occupied_history = [copy(occupied)]
	max_sugar_capacities = copy(board)
	for t in 2:(params.n_timesteps)
		reset_gradient!(
			board, agents, occupied, board_history, x_history, y_history,
			wealth_history, alive_history, occupied_history, t, params.gradient_horizon)
		abm_step!(board, agents, occupied, params.moving_rule,
			params.sugar_regeneration_rate, max_sugar_capacities)
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

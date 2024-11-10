export AxtellFirmsParams, RandomAxtellAgentInitializer

struct Agent{T}
    id::Int64
    theta::T
    effort::Vector{T}  # Now a single-element vector
    firm_id::Vector{Int64}  # Now a single-element vector
    neighbors::Vector{Int64}
end

struct Firm{T}
    id::Int64
    output::Vector{T}  # Now a single-element vector
    size::Vector{T}  # Now a single-element vector
end

abstract type UtilityFunction end

struct CobbDouglasUtility <: UtilityFunction end

get_type(utility::CobbDouglasUtility) = Float64

function (utility::CobbDouglasUtility)(
    firms_output, firms_size, theta_agent, effort_agent)
    if ignore_gradient(firms_output) == 0.0
        return 0.0
    end
    return (firms_output / firms_size)^theta_agent *
           (1.0 - effort_agent)^(1.0 - theta_agent)
end

function compute_group_effort_from_output(firm_output, agent_effort, a, b)
    """
    solves the equation output = a * (E + e) + b (E + e)^2 where E is the group effort (except the agent's one) and e is the individual effort.
    this expands to
    bx^2 + x * (a+2be) + a*e + b*e^2 - output = 0
    """
    A = b
    B = a + 2b * agent_effort
    C = a * agent_effort + b * agent_effort^2 - firm_output
    # solve the quadratic equation since A>0 take the positive root
    x = (-B + sqrt(B^2 - 4A * C + 1e-10)) / (2A) # add a small number to avoid NaNs
    return x
end

function compute_optimal_effort(theta_i, E_tilde_i, a, b)
    numerator = -a - 2b * (E_tilde_i - theta_i)
    numerator += sqrt(a^2 + 4 * a * b * theta_i^2 * (1 + E_tilde_i) +
                      4 * b^2 * theta_i^2 * (1 + E_tilde_i)^2)
    denominator = 2b * (1 + theta_i)
    #is_negative = ignore_gradient(numerator) < 0.0
    is_negative = differentiable_is_less(GaussianSmoothing(0.35), numerator, zero(typeof(numerator)))
    #is_negative = differentiable_is_less(SigmoidSmoothing(0.1), numerator, zero(typeof(numerator)))
    #is_negative = differentiable_is_less(PiecewiseSmoothing([-2.0, 2.0]), numerator, zero(typeof(numerator)))
    return (numerator / denominator) * (1.0 - is_negative)
end

function compute_firms_output(group_effort, a, b)
    return a * group_effort + b * group_effort^2
end

abstract type AbstractAxtellAgentInitializer end
struct GeneratedAxtellAgentInitializer{T} <: AbstractAxtellAgentInitializer
    thetas::Vector{T}
    efforts::Vector{T}
    neighbors::Vector{Vector{Int64}}
end
@functor GeneratedAxtellAgentInitializer (thetas, efforts)
get_type(initializer::GeneratedAxtellAgentInitializer{T}) where {T} = T

function initialize(initializer::GeneratedAxtellAgentInitializer{T}, diff_type) where {T}
    agents = Vector{Agent{diff_type}}()
    agents = [
        Agent(i, initializer.thetas[i], [initializer.efforts[i]], [i], initializer.neighbors[i])
        for i in 1:length(initializer.thetas)
    ]
    firms = Dict(i => Firm(i, [one(diff_type)], [one(diff_type)]) for i in 1:length(initializer.thetas))
    return agents, firms
end

struct RandomAxtellAgentInitializer{T,Q} <: AbstractAxtellAgentInitializer
    n_agents::Int64
    thetas_bounds::Vector{T}
    initial_efforts_bounds::Vector{Q}
    neighbours::Vector{Vector{Int64}}
end
@functor RandomAxtellAgentInitializer (thetas_bounds, initial_efforts_bounds)
get_type(initializer::RandomAxtellAgentInitializer{T}) where {T} = T
function initialize(initializer::RandomAxtellAgentInitializer{T}, diff_type) where {T}
    agents = Vector{Agent{diff_type}}()
    for i in 1:(initializer.n_agents)
        theta = convert(diff_type,
            rand(Beta(initializer.thetas_bounds[1], initializer.thetas_bounds[2])))
        # avoid NaNs
        eps = 1e-10
        theta = @. min(max(theta, eps), 1.0 - eps)
        effort = [convert(diff_type,
            rand(Beta(initializer.initial_efforts_bounds[1],
                initializer.initial_efforts_bounds[2])))]
        effort = @. min(max(effort, eps), 1.0 - eps)
        neighbors = initializer.neighbours[i]
        agent = Agent(i, theta, effort, [i], neighbors)
        push!(agents, agent)
    end
    firms = Dict(i => Firm(i, [one(diff_type)], [one(diff_type)]) for i in 1:initializer.n_agents)
    return agents, firms
end

function compute_current_firm_utility(
    agent::Agent{T}, current_firm::Firm{T}, utility_function::UtilityFunction, a, b) where {T}
    E_tilde_i = ignore_gradient.(compute_group_effort_from_output(
        current_firm.output[1], agent.effort[1], a, b))
    optimal_effort = compute_optimal_effort(agent.theta, E_tilde_i, a, b)
    stay_utility = utility_function(current_firm.output[1], current_firm.size[1],
        agent.theta, optimal_effort)
    return stay_utility, current_firm.id, optimal_effort
end

function compute_startup_utility(
    agent::Agent{T}, utility_function::UtilityFunction, a, b) where {T}
    new_firm_id = rand(1:Int64(1e10))
    new_firm_output = compute_firms_output(agent.effort[1], a, b)
    optimal_effort = compute_optimal_effort(agent.theta, zero(T), a, b)
    startup_utility = utility_function(
        new_firm_output, one(T), agent.theta, optimal_effort)
    return startup_utility, new_firm_id, optimal_effort
end

function compute_neighbor_utilities(
    agent::Agent{T}, agents::Vector{Agent{T}}, firms::Dict{Int64,Firm{T}},
    utility_function::UtilityFunction, a, b) where {T}
    utilities = T[]
    firm_ids = T[]
    optimal_efforts = T[]

    for neighbor_id in agent.neighbors
        neighbor_firm = firms[agents[neighbor_id].firm_id[1]]
        E_tilde_i = compute_group_effort_from_output(
            neighbor_firm.output[1], agent.effort[1], a, b)
        optimal_effort = compute_optimal_effort(
            agent.theta, E_tilde_i, a, b)
        switch_utility = utility_function(neighbor_firm.output[1], neighbor_firm.size[1],
            agent.theta, optimal_effort)

        push!(utilities, switch_utility)
        push!(firm_ids, neighbor_firm.id)
        push!(optimal_efforts, optimal_effort)
    end
    return utilities, firm_ids, optimal_efforts
end

function compute_agent_utilities(agents::Vector{Agent{T}}, agent::Agent{T},
    firms::Dict{Int64,Firm{T}}, utility_function::UtilityFunction, a, b) where {T}

    # Current firm
    current_utility, current_firm_id, current_optimal_effort = compute_current_firm_utility(
        agent, firms[agent.firm_id[1]], utility_function, a, b)

    # New firm
    startup_utility, startup_firm_id, startup_optimal_effort = compute_startup_utility(
        agent, utility_function, a, b)

    # Neighbor firms
    neighbor_utilities, neighbor_firm_ids, neighbor_optimal_efforts = compute_neighbor_utilities(
        agent, agents, firms, utility_function, a, b)

    utilities = vcat(current_utility, startup_utility, neighbor_utilities)
    firm_candidates_ids = vcat(current_firm_id, startup_firm_id, neighbor_firm_ids)
    optimal_efforts = vcat(
        current_optimal_effort, startup_optimal_effort, neighbor_optimal_efforts)
    return utilities, firm_candidates_ids, optimal_efforts
end

function update_firm_outputs!(
    firms::Dict{Int64,Firm{T}}, agents::Vector{Agent{T}}, a, b) where {T}
    # Update firm outputs
    group_efforts = DefaultDict{Int64,T}(zero(T))
    for agent in agents
        firm_id = agent.firm_id[1]
        group_efforts[firm_id] += agent.effort[1]
    end
    for key in keys(group_efforts)
        firms[key].output[1] = compute_firms_output(group_efforts[key], a, b)
    end
end

function step!(agents::Vector{Agent{T}}, firms::Dict{Int64,Firm{T}},
    utility_function::UtilityFunction, a, b, activation_rate, delta_t) where {T}
    activation_probability = 1.0 - exp(-activation_rate * delta_t)
    for agent in agents
        if rand() < activation_probability
            utilities, firm_candidates_ids, optimal_efforts = compute_agent_utilities(
                agents, agent, firms, utility_function, a, b)
            chosen_index_onehot = differentiable_argmax(utilities)
            # remove agent from old firm
            old_firm_id = agent.firm_id[1]
            firms[old_firm_id].size[1] -= 1
            new_firm_id = Int64(round(sum(firm_candidates_ids .* chosen_index_onehot)))
            # create new firm if necessary
            if new_firm_id ∉ keys(firms)
                firms[new_firm_id] = Firm(new_firm_id, [zero(T)], [zero(T)])
            end
            # add aggent to new firm
            for (i, candidate_firm_id) in enumerate(firm_candidates_ids)
                if candidate_firm_id ∈ keys(firms)
                    firms[candidate_firm_id].size[1] += chosen_index_onehot[i]
                end
            end
            # Update agent effort
            agent.effort[1] = sum(optimal_efforts .* chosen_index_onehot)
            agent.firm_id[1] = new_firm_id
        end
    end
    update_firm_outputs!(firms, agents, a, b)
end

struct AxtellFirmsParams{T}
    agent_initializer::AbstractAxtellAgentInitializer
    a::Vector{T}
    b::Vector{T}
    activation_rate::Float64
    delta_t::Float64
    n_steps::Int64
    gradient_horizon::Int64
end
@functor AxtellFirmsParams (agent_initializer, a, b)

function reconstruct_firms_agents_no_gradient(
    firms::Dict{Int64,Firm{T}}, agents::Vector{Agent{T}}) where {T}
    firms = Dict(firm.id => Firm(firm.id, convert.(T, ignore_gradient.(firm.output)), convert.(T, ignore_gradient.(firm.size))) for firm in values(firms))
    agents = [Agent(agent.id,
        agent.theta,
        convert.(T, ignore_gradient.(agent.effort)),
        agent.firm_id,
        agent.neighbors) for agent in agents]
    return firms, agents
end

function abm_run(params::AxtellFirmsParams)
    utility_function = CobbDouglasUtility()
    diff_type = promote_type(
        get_type(params.agent_initializer), typeof(params.a[1]), typeof(params.b[1]), get_type(utility_function))
    agents, firms = initialize(params.agent_initializer, diff_type)
    a = convert(diff_type, params.a[1])
    b = convert(diff_type, params.b[1])
    update_firm_outputs!(firms, agents, a, b)
    mean_effort_by_timestep = [mean([agent.effort[1] for agent in agents])]
    mean_firm_output_by_timestep = [mean([firm.output[1] for firm in values(firms) if firm.size[1] > 0])]
    mean_firm_size_by_timestep = [mean([firm.size[1] for firm in values(firms) if firm.size[1] > 0])]
    for t in 2:(params.n_steps)
        if t % params.gradient_horizon == 0
            firms, agents = reconstruct_firms_agents_no_gradient(firms, agents)
        end
        step!(agents, firms, utility_function, params.a[1],
            params.b[1], params.activation_rate, params.delta_t)
        push!(mean_effort_by_timestep, mean([agent.effort[1] for agent in agents]))
        push!(mean_firm_output_by_timestep, mean([firm.output[1] for firm in values(firms) if firm.size[1] > 0]))
        push!(mean_firm_size_by_timestep,
            mean([firm.size[1] for firm in values(firms) if firm.size[1] > 0]))
    end
    return hcat(mean_effort_by_timestep, mean_firm_size_by_timestep, mean_firm_output_by_timestep)'
end

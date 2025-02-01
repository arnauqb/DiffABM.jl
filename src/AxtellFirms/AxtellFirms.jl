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
    a::T
    b::T
end

struct DiffBeta{T} <: Distributions.ContinuousUnivariateDistribution
    alpha::T
    beta::T
end
function Distributions.rand(rng::AbstractRNG, dist::DiffBeta{T}, n_samples::Int64) where {T}
    # sample from Kumaraswamy distribution as a quick diff approximation
    u = rand(rng, n_samples)
    return (1 .- u .^ (1 / dist.beta)) .^ (1 / dist.alpha)
end
function Distributions.rand(rng::AbstractRNG, dist::DiffBeta{T}) where {T}
    u = rand(rng)
    return (1 .- u .^ (1 / dist.beta)) .^ (1 / dist.alpha)
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

function compute_group_effort_from_output(firm_output, agent_effort, firm_a, firm_b)
    """
    solves the equation output = a * (E + e) + b (E + e)^2 where E is the group effort (except the agent's one) and e is the individual effort.
    this expands to
    bx^2 + x * (a+2be) + a*e + b*e^2 - output = 0
    """
    A = firm_b
    B = firm_a + 2 * firm_b * agent_effort
    C = firm_a * agent_effort + firm_b * agent_effort^2 - firm_output
    # solve the quadratic equation since A>0 take the positive root
    x = (-B + sqrt(B^2 - 4A * C + 1e-10)) / (2A) # add a small number to avoid NaNs
    return x
end

function compute_optimal_effort(theta_i, E_tilde_i, firm_a, firm_b)
    numerator = -firm_a - 2 * firm_b * (E_tilde_i - theta_i)
    numerator += sqrt(firm_a^2 + 4 * firm_a * firm_b * theta_i^2 * (1 + E_tilde_i) +
                      4 * firm_b^2 * theta_i^2 * (1 + E_tilde_i)^2)
    denominator = 2 * firm_b * (1 + theta_i)
    #is_negative = differentiable_is_less(SigmoidSmoothing(10.0), numerator, zero(typeof(numerator)))
    #is_negative = numerator < 0
    #return (numerator / denominator) * (1.0 - is_negative)
    return max(numerator / denominator, 0.0)
end

function compute_firms_output(group_effort, firm_a, firm_b)
    return firm_a * group_effort + firm_b * group_effort^2
end

abstract type AbstractAxtellAgentInitializer end
struct GeneratedAxtellAgentInitializer{T} <: AbstractAxtellAgentInitializer
    thetas::Vector{T}
    efforts::Vector{T}
    neighbors::Vector{Vector{Int64}}
end
@functor GeneratedAxtellAgentInitializer (thetas, efforts)
get_type(initializer::GeneratedAxtellAgentInitializer{T}) where {T} = T

function initialize(initializer::GeneratedAxtellAgentInitializer{T}, firm_a_generator, firm_b_generator, diff_type) where {T}
    agents = Vector{Agent{diff_type}}()
    agents = [
        Agent(i, initializer.thetas[i], [initializer.efforts[i]], [i], initializer.neighbors[i])
        for i in 1:length(initializer.thetas)
    ]
    firms = Dict(i => Firm(i, [one(diff_type)], [one(diff_type)], 2 * rand(firm_a_generator), 2 * rand(firm_b_generator)) for i in 1:length(initializer.thetas))
    return agents, firms, firm_a_generator, firm_b_generator
end

struct RandomAxtellAgentInitializer{T,Q,F} <: AbstractAxtellAgentInitializer
    n_agents::Int64
    thetas_bounds::Vector{T}
    initial_efforts_bounds::Vector{Q}
    a_bounds::Vector{F}
    b_bounds::Vector{F}
    n_neighbours::Int64
end
@functor RandomAxtellAgentInitializer (thetas_bounds, initial_efforts_bounds, a_bounds, b_bounds)
get_type(initializer::RandomAxtellAgentInitializer{T,Q,F}) where {T,Q,F} = promote_type(T, Q, F)
function initialize(initializer::RandomAxtellAgentInitializer{T}, diff_type) where {T}
    agents = Vector{Agent{diff_type}}()
    theta_generator = DiffBeta(initializer.thetas_bounds[1], initializer.thetas_bounds[2])
    effort_generator = DiffBeta(initializer.initial_efforts_bounds[1], initializer.initial_efforts_bounds[2])
    a_generator = DiffBeta(initializer.a_bounds[1], initializer.a_bounds[2])
    b_generator = DiffBeta(initializer.b_bounds[1], initializer.b_bounds[2])
    thetas = rand(theta_generator, initializer.n_agents)
    efforts = rand(effort_generator, initializer.n_agents)
    for i in 1:(initializer.n_agents)
        theta = thetas[i]
        effort = [efforts[i]]
        neighbors = sample(1:initializer.n_agents, initializer.n_neighbours, replace=false)
        agent = Agent(i, theta, effort, [i], neighbors)
        push!(agents, agent)
    end
    firms = Dict(i => Firm(i, [one(diff_type)], [one(diff_type)], 2 * rand(a_generator), 2 * rand(b_generator)) for i in 1:initializer.n_agents)
    return agents, firms, a_generator, b_generator
end

function compute_current_firm_utility(
    agent::Agent{T}, current_firm::Firm{T}, utility_function::UtilityFunction) where {T}
    E_tilde_i = ignore_gradient.(compute_group_effort_from_output(
        current_firm.output[1], agent.effort[1], current_firm.a, current_firm.b))
    optimal_effort = compute_optimal_effort(agent.theta, E_tilde_i, current_firm.a, current_firm.b)
    stay_utility = utility_function(current_firm.output[1], current_firm.size[1],
        agent.theta, optimal_effort)
    return stay_utility, current_firm.id, optimal_effort
end

function compute_startup_utility(
    agent::Agent{T}, utility_function::UtilityFunction, new_firm_a, new_firm_b) where {T}
    new_firm_id = rand(1:Int64(1e10))
    new_firm_output = compute_firms_output(agent.effort[1], new_firm_a, new_firm_b)
    optimal_effort = compute_optimal_effort(agent.theta, zero(T), new_firm_a, new_firm_b)
    startup_utility = utility_function(
        new_firm_output, one(T), agent.theta, optimal_effort)
    return startup_utility, new_firm_id, optimal_effort
end

function compute_neighbor_utilities(
    agent::Agent{T}, agents::Vector{Agent{T}}, firms::Dict{Int64,Firm{T}},
    utility_function::UtilityFunction) where {T}
    utilities = T[]
    firm_ids = T[]
    optimal_efforts = T[]

    for neighbor_id in agent.neighbors
        neighbor_firm = firms[agents[neighbor_id].firm_id[1]]
        E_tilde_i = compute_group_effort_from_output(
            neighbor_firm.output[1], agent.effort[1], neighbor_firm.a, neighbor_firm.b)
        optimal_effort = compute_optimal_effort(
            agent.theta, E_tilde_i, neighbor_firm.a, neighbor_firm.b)
        switch_utility = utility_function(neighbor_firm.output[1], neighbor_firm.size[1],
            agent.theta, optimal_effort)

        push!(utilities, switch_utility)
        push!(firm_ids, neighbor_firm.id)
        push!(optimal_efforts, optimal_effort)
    end
    return utilities, firm_ids, optimal_efforts
end

function compute_agent_utilities(agents::Vector{Agent{T}}, agent::Agent{T},
    firms::Dict{Int64,Firm{T}}, utility_function::UtilityFunction, firm_a_generator, firm_b_generator) where {T}

    # Current firm
    current_utility, current_firm_id, current_optimal_effort = compute_current_firm_utility(
        agent, firms[agent.firm_id[1]], utility_function)

    # New firm
    new_firm_a = 2 * rand(firm_a_generator)
    new_firm_b = 2 * rand(firm_b_generator)
    startup_utility, startup_firm_id, startup_optimal_effort = compute_startup_utility(
        agent, utility_function, new_firm_a, new_firm_b)

    # Neighbor firms
    neighbor_utilities, neighbor_firm_ids, neighbor_optimal_efforts = compute_neighbor_utilities(
        agent, agents, firms, utility_function)

    utilities = vcat(current_utility, startup_utility, neighbor_utilities)
    firm_candidates_ids = vcat(current_firm_id, startup_firm_id, neighbor_firm_ids)
    optimal_efforts = vcat(
        current_optimal_effort, startup_optimal_effort, neighbor_optimal_efforts)
    return utilities, firm_candidates_ids, optimal_efforts, new_firm_a, new_firm_b
end

function update_firm_outputs!(
    firms::Dict{Int64,Firm{T}}, agents::Vector{Agent{T}}) where {T}
    # Update firm outputs
    group_efforts = DefaultDict{Int64,T}(zero(T))
    for agent in agents
        firm_id = agent.firm_id[1]
        group_efforts[firm_id] += agent.effort[1]
    end
    for key in keys(group_efforts)
        firms[key].output[1] = compute_firms_output(group_efforts[key], firms[key].a, firms[key].b)
    end
end

function step!(agents::Vector{Agent{T}}, firms::Dict{Int64,Firm{T}},
    utility_function::UtilityFunction, firm_a_generator, firm_b_generator, activation_rate, delta_t) where {T}
    activation_probability = 1.0 - exp(-activation_rate * delta_t)
    for agent in agents
        if rand() < activation_probability
            utilities, firm_candidates_ids, optimal_efforts, new_firm_a, new_firm_b = compute_agent_utilities(
                agents, agent, firms, utility_function, firm_a_generator, firm_b_generator)
            chosen_index_onehot = differentiable_argmax(utilities)
            # remove agent from old firm
            old_firm_id = agent.firm_id[1]
            firms[old_firm_id].size[1] -= 1
            new_firm_id = Int64(round(sum(firm_candidates_ids .* chosen_index_onehot)))
            # create new firm if necessary
            if new_firm_id ∉ keys(firms)
                firms[new_firm_id] = Firm(new_firm_id, [zero(T)], [zero(T)], new_firm_a, new_firm_b)
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
    update_firm_outputs!(firms, agents)
end

struct AxtellFirmsParams
    agent_initializer::AbstractAxtellAgentInitializer
    activation_rate::Float64
    delta_t::Float64
    n_steps::Int64
    gradient_horizon::Int64
end
@functor AxtellFirmsParams (agent_initializer,)

function abm_run(params::AxtellFirmsParams)
    utility_function = CobbDouglasUtility()
    diff_type = promote_type(
        get_type(params.agent_initializer), get_type(utility_function))
    agents, firms, firm_a_generator, firm_b_generator = initialize(params.agent_initializer, diff_type)
    update_firm_outputs!(firms, agents)
    mean_effort_by_timestep = [mean([agent.effort[1] for agent in agents])]
    mean_firm_output_by_timestep = [mean([firm.output[1] for firm in values(firms) if firm.size[1] > 0])]
    mean_firm_size_by_timestep = [mean([firm.size[1] for firm in values(firms) if firm.size[1] > 0])]
    for t in 2:(params.n_steps)
        step!(agents, firms, utility_function, firm_a_generator, firm_b_generator, params.activation_rate, params.delta_t)
        push!(mean_effort_by_timestep, mean([agent.effort[1] for agent in agents]))
        push!(mean_firm_output_by_timestep, mean([firm.output[1] for firm in values(firms) if firm.size[1] > 0]))
        push!(mean_firm_size_by_timestep,
            mean([firm.size[1] for firm in values(firms) if firm.size[1] > 0]))
    end
    return hcat(mean_effort_by_timestep, mean_firm_size_by_timestep, mean_firm_output_by_timestep)'
end

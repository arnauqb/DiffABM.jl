export AxtellFirmsParams

struct Agent{T}
    id::Int64
    theta::T
    endowment::T
    effort::Vector{T}  # Now a single-element vector
    firm_id::Vector{T}  # Now a single-element vector
    neighbors::Vector{Int64}
end

struct Firm{T}
    id::T
    output::Vector{T}  # Now a single-element vector
    size::Vector{T}  # Now a single-element vector
end

function compute_group_effort_from_output(output, effort, a, b)
    """
    solves the equation output = a * (E + e) + b (E + e)^2 where E is the group effort (except the agent's one) and e is the individual effort.
    this expands to
    bx^2 + x * (a+2be) + a*e + b*e^2 - output = 0
    """
    A = b
    B = a + 2b * effort
    C = a * effort + b * effort^2 - output
    # solve the quadratic equation since A>0 take the positive root
    x = (-B + sqrt(B^2 - 4A * C)) / (2A)
    return x
end

function compute_optimal_effort(theta_i, omega_i, E_tilde_i, a, b)
    numerator = -a -2b*(E_tilde_i - theta_i)
    numerator += sqrt(a^2 + 4*a*b*theta_i^2*(1+E_tilde_i) + 4*b^2*theta_i^2*(1+E_tilde_i)^2)
    denominator = 2b * (1 + theta_i)
    return max(numerator / denominator, zero(typeof(numerator)))
end

function compute_utility(
        firms_output, firms_size, theta_agent, endowment_agent, effort_agent)
    if ignore_gradient(firms_output) == 0.0
        return 0.0
    end
    return (firms_output / firms_size)^theta_agent *
           (endowment_agent - effort_agent)^(1.0 - theta_agent)
end

function compute_firms_output(group_effort, a, b)
    return a * group_effort + b * group_effort^2
end

abstract type AbstractAxtellAgentInitializer end
struct GeneratedAxtellAgentInitializer{T} <: AbstractAxtellAgentInitializer
    thetas::Vector{T}
    endowments::Vector{T}
    efforts::Vector{T}
    neighbors::Vector{Vector{Int64}}
end
get_type(initializer::GeneratedAxtellAgentInitializer{T}) where {T} = T

function initialize(initializer::GeneratedAxtellAgentInitializer{T}, diff_type) where {T}
    agents = Vector{Agent{diff_type}}()
    firms = Vector{Firm{diff_type}}()

    for i in 1:length(initializer.thetas)
        theta = initializer.thetas[i]
        endowment = initializer.endowments[i]
        effort = [initializer.efforts[i]]
        neighbors = initializer.neighbors[i]

        push!(
            agents, Agent(i, theta, endowment, effort, [convert(diff_type, i)], neighbors))
        push!(firms, Firm(convert(diff_type, i), [zero(diff_type)], [one(diff_type)]))
    end
    return agents, firms
end

struct RandomAxtellAgentInitializer{T} <: AbstractAxtellAgentInitializer
    n_agents::Int64
    thetas_bounds::Tuple{T, T}
    endowments_bounds::Tuple{T, T}
    initial_efforts_bounds::Tuple{T, T}
    neighbours::Vector{Vector{Int64}}
end
get_type(initializer::RandomAxtellAgentInitializer{T}) where {T} = T
function initialize(initializer::RandomAxtellAgentInitializer{T}, diff_type) where {T}
    agents = Vector{Agent{diff_type}}()
    firms = Vector{Firm{diff_type}}()
    for i in 1:(initializer.n_agents)
        theta = convert(diff_type, initializer.thetas_bounds[1] +
                (initializer.thetas_bounds[2] - initializer.thetas_bounds[1]) * rand())
        endowment = convert(diff_type, 1.0)
        effort = initializer.initial_efforts_bounds[1] +
                  (initializer.initial_efforts_bounds[2] -
                   initializer.initial_efforts_bounds[1]) * rand()
        effort = [convert(diff_type, effort)]
        neighbors = initializer.neighbours[i]
        agent = Agent(i, theta, endowment, effort, [convert(diff_type, i)], neighbors)
        push!(agents, agent)
        push!(firms, Firm(convert(diff_type, i), [one(diff_type)], [one(diff_type)]))
    end
    return agents, firms
end

function compute_agent_utilities(
        agents, agent::Agent{T}, firms::Vector{Firm{T}}, a::T, b::T) where {T}
    utilities = T[]
    firm_candidates_ids = T[]
    optimal_efforts = T[]

    # Current firm
    current_firm = firms[agent.firm_id[1]]
    E_tilde_i = compute_group_effort_from_output(
        current_firm.output[1], agent.effort[1], a, b)
    optimal_effort = compute_optimal_effort(agent.theta, agent.endowment, E_tilde_i, a, b)
    stay_utility = compute_utility(current_firm.output[1], current_firm.size[1],
        agent.theta, agent.endowment, optimal_effort)

    push!(utilities, stay_utility)
    push!(firm_candidates_ids, agent.firm_id[1])
    push!(optimal_efforts, optimal_effort)

    # New firm
    new_firm_output = compute_firms_output(agent.effort[1], a, b)
    optimal_effort = compute_optimal_effort(agent.theta, agent.endowment, zero(T), a, b)
    startup_utility = compute_utility(
        new_firm_output, one(T), agent.theta, agent.endowment, optimal_effort)

    push!(utilities, startup_utility)
    push!(firm_candidates_ids, length(firms) + 1)
    push!(optimal_efforts, optimal_effort)

    # Neighbor firms
    for neighbor_id in agent.neighbors
        neighbor_firm = firms[agents[neighbor_id].firm_id[1]]
        E_tilde_i = compute_group_effort_from_output(
            neighbor_firm.output[1], agent.effort[1], a, b)
        optimal_effort = compute_optimal_effort(
            agent.theta, agent.endowment, E_tilde_i, a, b)
        switch_utility = compute_utility(neighbor_firm.output[1], neighbor_firm.size[1],
            agent.theta, agent.endowment, optimal_effort)

        push!(utilities, switch_utility)
        push!(firm_candidates_ids, neighbor_firm.id)
        push!(optimal_efforts, optimal_effort)
    end
    return utilities, firm_candidates_ids, optimal_efforts
end

function update_firm_outputs!(firms::Vector{Firm{T}}, agents::Vector{Agent{T}}, a::T, b::T) where {T}
    # Update firm outputs
    group_efforts = zeros(T, length(firms))
    for agent in agents
        firm = firms[agent.firm_id[1]]
        group_efforts[firm.id] += agent.effort[1]
    end
    for (i, firm) in enumerate(firms)
        firm.output[1] = compute_firms_output(group_efforts[i], a, b)
    end
end

function step!(agents::Vector{Agent{T}}, firms::Vector{Firm{T}}, a::T, b::T, activation_rate, delta_t) where {T}
    activation_probability = 1.0 - exp(-activation_rate * delta_t)
    for agent in agents
        if rand() < activation_probability
            utilities, firm_candidates_ids, optimal_efforts = compute_agent_utilities(
                agents, agent, firms, a, b)
            chosen_index_onehot = differentiable_argmax(utilities)
            # remove agent from old firm
            old_firm = firms[agent.firm_id[1]]
            old_firm.size[1] -= 1
            # add agent to new firm
            new_firm_id = zero(T)
            new_effort = zero(T)
            for (i, candidate_firm_id) in enumerate(firm_candidates_ids)
                new_firm_id += candidate_firm_id * chosen_index_onehot[i]
                new_effort += optimal_efforts[i] * chosen_index_onehot[i]
            end
            if new_firm_id > length(firms)
                push!(firms, Firm(new_firm_id, [one(T)], [one(T)]))
                firms[new_firm_id].size[1] += 1
            else
                firms[new_firm_id].size[1] += 1
            end

            # Update agent
            agent.effort[1] = new_effort
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
    soft_histogram_sigma::Float64
end
@functor AxtellFirmsParams (agent_initializer, a, b)

function abm_run(params::AxtellFirmsParams)
    diff_type = promote_type(
        get_type(params.agent_initializer), typeof(params.a[1]), typeof(params.b[1]))
    agents, firms = initialize(params.agent_initializer, diff_type)
    update_firm_outputs!(firms, agents, params.a[1], params.b[1])
    mean_effort_by_timestep = [mean([agent.effort[1] for agent in agents])]
    mean_firm_output_by_timestep = [mean([firm.output[1] for firm in firms])]
    for t in 2:(params.n_steps)
        step!(agents, firms, params.a[1], params.b[1], params.activation_rate, params.delta_t)
        push!(mean_effort_by_timestep, mean([agent.effort[1] for agent in agents]))
        push!(mean_firm_output_by_timestep, mean([firm.output[1] for firm in firms]))
    end
    return hcat(mean_effort_by_timestep, mean_firm_output_by_timestep)
end

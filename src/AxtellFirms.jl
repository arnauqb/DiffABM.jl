export AxtellFirmsParams, CobbDouglasUtility, NeuralNetworkUtility

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

abstract type UtilityFunction end
struct CobbDouglasUtility <: UtilityFunction end
get_type(utility::CobbDouglasUtility) = Float64
struct NeuralNetworkUtility{T} <: UtilityFunction
    nn::T
end
get_type(utility::NeuralNetworkUtility) = typeof(utility.nn.layers[1].weight[1])
@functor NeuralNetworkUtility (nn,)
function (utility::CobbDouglasUtility)(
    firms_output, firms_size, theta_agent, endowment_agent, effort_agent)
    if ignore_gradient(firms_output) == 0.0
        return 0.0
    end
    return (firms_output / firms_size)^theta_agent *
           (endowment_agent - effort_agent)^(1.0 - theta_agent)
end
function (utility::NeuralNetworkUtility)(
    firms_output, firms_size, theta_agent, endowment_agent, effort_agent)
    return utility.nn([firms_output, firms_size, theta_agent, endowment_agent, effort_agent])[1]
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
    x = (-B + sqrt(B^2 - 4A * C + 1e-8)) / (2A) # add a small number to avoid NaNs
    return x
end

function compute_optimal_effort(theta_i, omega_i, E_tilde_i, a, b)
    numerator = -a - 2b * (E_tilde_i - theta_i)
    numerator += sqrt(a^2 + 4 * a * b * theta_i^2 * (1 + E_tilde_i) + 4 * b^2 * theta_i^2 * (1 + E_tilde_i)^2)
    denominator = 2b * (1 + theta_i)
    return max(numerator / denominator, zero(typeof(numerator)))
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
    thetas_bounds::Vector{T}
    initial_efforts_bounds::Vector{T}
    neighbours::Vector{Vector{Int64}}
end
@functor RandomAxtellAgentInitializer (thetas_bounds, initial_efforts_bounds)
get_type(initializer::RandomAxtellAgentInitializer{T}) where {T} = T
function initialize(initializer::RandomAxtellAgentInitializer{T}, diff_type) where {T}
    agents = Vector{Agent{diff_type}}()
    firms = Vector{Firm{diff_type}}()
    for i in 1:(initializer.n_agents)
        theta = convert(diff_type, initializer.thetas_bounds[1] +
                                   (initializer.thetas_bounds[2] - initializer.thetas_bounds[1]) * rand())
        #endowment = convert(diff_type, initializer.endowments_bounds[1] +
        #                     (initializer.endowments_bounds[2] - initializer.endowments_bounds[1]) * rand())
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
    agents, agent::Agent{T}, firms::Vector{Firm{T}}, utility_function::UtilityFunction, a, b) where {T}
    utilities = T[]
    firm_candidates_ids = T[]
    optimal_efforts = T[]

    # Current firm
    current_firm = firms[agent.firm_id[1]]
    E_tilde_i = compute_group_effort_from_output(
        current_firm.output[1], agent.effort[1], a, b)
    optimal_effort = compute_optimal_effort(agent.theta, agent.endowment, E_tilde_i, a, b)
    stay_utility = utility_function(current_firm.output[1], current_firm.size[1],
        agent.theta, agent.endowment, optimal_effort)

    push!(utilities, stay_utility)
    push!(firm_candidates_ids, agent.firm_id[1])
    push!(optimal_efforts, optimal_effort)

    # New firm
    new_firm_output = compute_firms_output(agent.effort[1], a, b)
    optimal_effort = compute_optimal_effort(agent.theta, agent.endowment, zero(T), a, b)
    startup_utility = utility_function(
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
        switch_utility = utility_function(neighbor_firm.output[1], neighbor_firm.size[1],
            agent.theta, agent.endowment, optimal_effort)

        push!(utilities, switch_utility)
        push!(firm_candidates_ids, neighbor_firm.id)
        push!(optimal_efforts, optimal_effort)
    end
    return utilities, firm_candidates_ids, optimal_efforts
end

function update_firm_outputs!(firms::Vector{Firm{T}}, agents::Vector{Agent{T}}, a, b) where {T}
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

function step!(agents::Vector{Agent{T}}, firms::Vector{Firm{T}}, utility_function::UtilityFunction, a, b, activation_rate, delta_t) where {T}
    activation_probability = 1.0 - exp(-activation_rate * delta_t)
    for agent in agents
        if rand() < activation_probability
            utilities, firm_candidates_ids, optimal_efforts = compute_agent_utilities(
                agents, agent, firms, utility_function, a, b)
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
    utility_function::UtilityFunction
    a::Vector{T}
    b::Vector{T}
    activation_rate::Float64
    delta_t::Float64
    n_steps::Int64
    gradient_horizon::Int64
end
@functor AxtellFirmsParams (agent_initializer, utility_function, a, b)

function reconstruct_firms_agents_no_gradient(firms::Vector{Firm{T}}, agents::Vector{Agent{T}}) where {T}
    firms = [Firm(firm.id, convert.(T, ignore_gradient.(firm.output)), convert.(T, ignore_gradient.(firm.size))) for firm in firms]
    agents = [Agent(agent.id,
        agent.theta,
        agent.endowment,
        convert.(T, ignore_gradient.(agent.effort)),
        convert.(T, ignore_gradient.(agent.firm_id)),
        agent.neighbors) for agent in agents]
    return firms, agents
end

function abm_run(params::AxtellFirmsParams)
    diff_type = promote_type(
        get_type(params.agent_initializer), typeof(params.a[1]), typeof(params.b[1]), get_type(params.utility_function))
    agents, firms = initialize(params.agent_initializer, diff_type)
    a = convert(diff_type, params.a[1])
    b = convert(diff_type, params.b[1])
    update_firm_outputs!(firms, agents, a, b)
    mean_effort_by_timestep = [mean([agent.effort[1] for agent in agents])]
    mean_firm_output_by_timestep = [mean([firm.output[1] for firm in firms])]
    mean_firm_size_by_timestep = [mean([firm.size[1] for firm in firms if firm.size[1] > 0])]
    for t in 2:(params.n_steps)
        if t % params.gradient_horizon == 0
            firms, agents = reconstruct_firms_agents_no_gradient(firms, agents)
        end
        step!(agents, firms, params.utility_function, params.a[1], params.b[1], params.activation_rate, params.delta_t)
        push!(mean_effort_by_timestep, mean([agent.effort[1] for agent in agents]))
        push!(mean_firm_output_by_timestep, mean([firm.output[1] for firm in firms]))
        push!(mean_firm_size_by_timestep, mean([firm.size[1] for firm in firms if firm.size[1] > 0]))
    end
    return hcat(mean_effort_by_timestep, mean_firm_size_by_timestep, mean_firm_output_by_timestep)'
    #return reshape(mean_firm_output_by_timestep, 1, :)
end

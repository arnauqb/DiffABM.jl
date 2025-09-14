export AxtellFirmsParams, RandomAxtellAgentInitializer

"""
    Agent{T}

Represents an economic agent in the Axtell Firms model.

Each agent is characterized by a skill parameter θ and makes discrete firm selection
decisions that are made differentiable through surrogate gradients. The agent chooses
among firms to maximize their Cobb-Douglas utility function.

# Fields
- `id::Int64`: Unique agent identifier
- `theta::T`: Skill parameter θ ∈ [0,1] determining production capability
- `effort::Vector{T}`: Current effort level (single-element vector for mutability)
- `firm_id::Vector{Int64}`: Current firm membership (single-element vector for mutability)
- `neighbors::Vector{Int64}`: List of neighboring agent IDs for firm discovery
"""
struct Agent{T}
    id::Int64
    theta::T
    effort::Vector{T}  # Now a single-element vector
    firm_id::Vector{Int64}  # Now a single-element vector
    neighbors::Vector{Int64}
end

"""
    Firm{T}

Represents a firm in the Axtell Firms model with quadratic production function.

Firms produce output based on the total effort of their employees according to:
Q = a·E + b·E², where E is total firm effort and a,b are production parameters.

# Fields
- `id::Int64`: Unique firm identifier
- `output::Vector{T}`: Current firm output (single-element vector for mutability)
- `size::Vector{T}`: Current firm size in number of employees (single-element vector for mutability)
- `a::T`: Linear production parameter
- `b::T`: Quadratic production parameter
"""
struct Firm{T}
    id::Int64
    output::Vector{T}  # Now a single-element vector
    size::Vector{T}  # Now a single-element vector
    a::T
    b::T
end

abstract type UtilityFunction end

"""
    CobbDouglasUtility <: UtilityFunction

Implements the Cobb-Douglas utility function for agents in the Axtell Firms model.

The utility function is: U = (Q/|f|)^θ · (1-e)^(1-θ)
where Q is firm output, |f| is firm size, θ is agent skill, and e is effort level.
"""
struct CobbDouglasUtility <: UtilityFunction end

get_type(utility::CobbDouglasUtility) = Float64

"""
    (utility::CobbDouglasUtility)(firms_output, firms_size, theta_agent, effort_agent)

Compute the Cobb-Douglas utility function for an agent.

The utility function is: U = (Q/|f|)^θ · (1-e)^(1-θ)
where Q is firm output, |f| is firm size, θ is agent skill parameter, and e is effort level.
This captures the trade-off between income (firm productivity) and leisure (1 - effort).

# Arguments
- `firms_output`: Total output Q of the firm
- `firms_size`: Size |f| of the firm (number of employees)
- `theta_agent`: Agent's skill parameter θ ∈ [0,1]
- `effort_agent`: Agent's effort level e ∈ [0,1]

# Returns
- `Float64`: Utility value
"""
function (utility::CobbDouglasUtility)(
        firms_output, firms_size, theta_agent, effort_agent)
    if ignore_gradient(firms_output) == 0.0
        return 0.0
    end
    return (firms_output / firms_size)^theta_agent *
           (1.0 - effort_agent)^(1.0 - theta_agent)
end

"""
    compute_group_effort_from_output(firm_output, agent_effort, firm_a, firm_b)

Solve for group effort E given firm output and production parameters.

Given the production function Q = a·(E + e) + b·(E + e)², where E is the group effort
(excluding the agent's effort e), this function solves for E by rearranging to:
bE² + E·(a + 2be) + ae + be² - Q = 0

Since b > 0, we take the positive root of the quadratic equation.

# Arguments
- `firm_output`: Current firm output Q
- `agent_effort`: Individual agent's effort e
- `firm_a`: Linear production parameter a
- `firm_b`: Quadratic production parameter b

# Returns
- `Float64`: Group effort E of other agents in the firm
"""
function compute_group_effort_from_output(firm_output, agent_effort, firm_a, firm_b)
    A = firm_b
    B = firm_a + 2 * firm_b * agent_effort
    C = firm_a * agent_effort + firm_b * agent_effort^2 - firm_output
    # solve the quadratic equation since A>0 take the positive root
    x = (-B + sqrt(B^2 - 4A * C + 1e-10)) / (2A) # add a small number to avoid NaNs
    return x
end

"""
    compute_optimal_effort(theta_i, E_tilde_i, firm_a, firm_b)

Compute optimal effort level for an agent given firm characteristics.

This function computes the effort level e* that maximizes the agent's Cobb-Douglas
utility function subject to the firm's quadratic production function. The optimization
leads to a closed-form solution involving the agent's skill θ and the group effort Ẽ.

# Arguments
- `theta_i`: Agent's skill parameter θᵢ ∈ [0,1]
- `E_tilde_i`: Group effort Ẽᵢ of other agents in the firm
- `firm_a`: Linear production parameter a
- `firm_b`: Quadratic production parameter b

# Returns
- `Float64`: Optimal effort level e* ∈ [0,1]
"""
function compute_optimal_effort(theta_i, E_tilde_i, firm_a, firm_b)
    numerator = -firm_a - 2 * firm_b * (E_tilde_i - theta_i)
    numerator += sqrt(firm_a^2 + 4 * firm_a * firm_b * theta_i^2 * (1 + E_tilde_i) +
                      4 * firm_b^2 * theta_i^2 * (1 + E_tilde_i)^2)
    denominator = 2 * firm_b * (1 + theta_i)
    return max(numerator / denominator, 0.0)
end

"""
    compute_firms_output(group_effort, firm_a, firm_b)

Compute firm output using the quadratic production function.

The production function is: Q = a·E + b·E²
where E is the total effort of all agents in the firm.

# Arguments
- `group_effort`: Total effort E of all agents in the firm
- `firm_a`: Linear production parameter a
- `firm_b`: Quadratic production parameter b

# Returns
- `Float64`: Firm output Q
"""
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

function initialize(initializer::GeneratedAxtellAgentInitializer{T},
        firm_a_generator, firm_b_generator, diff_type) where {T}
    agents = Vector{Agent{diff_type}}()
    agents = [Agent(i, initializer.thetas[i], [initializer.efforts[i]],
                  [i], initializer.neighbors[i])
              for i in 1:length(initializer.thetas)]
    firms = Dict(i => Firm(i, [one(diff_type)], [one(diff_type)],
                     2 * rand(firm_a_generator), 2 * rand(firm_b_generator))
    for i in 1:length(initializer.thetas))
    return agents, firms, firm_a_generator, firm_b_generator
end

struct RandomAxtellAgentInitializer{T, Q, F} <: AbstractAxtellAgentInitializer
    n_agents::Int64
    thetas_bounds::Vector{T}
    initial_efforts_bounds::Vector{Q}
    a_bounds::Vector{F}
    b_bounds::Vector{F}
    n_neighbours::Int64
end
@functor RandomAxtellAgentInitializer (
    thetas_bounds, initial_efforts_bounds, a_bounds, b_bounds)
function get_type(initializer::RandomAxtellAgentInitializer{T, Q, F}) where {T, Q, F}
    promote_type(T, Q, F)
end
function initialize(initializer::RandomAxtellAgentInitializer{T}, diff_type) where {T}
    agents = Vector{Agent{diff_type}}()
    theta_generator = DiffBeta(initializer.thetas_bounds[1], initializer.thetas_bounds[2])
    effort_generator = DiffBeta(
        initializer.initial_efforts_bounds[1], initializer.initial_efforts_bounds[2])
    a_generator = DiffBeta(initializer.a_bounds[1], initializer.a_bounds[2])
    b_generator = DiffBeta(initializer.b_bounds[1], initializer.b_bounds[2])
    thetas = rand(theta_generator, initializer.n_agents)
    efforts = rand(effort_generator, initializer.n_agents)
    for i in 1:(initializer.n_agents)
        theta = thetas[i]
        effort = [efforts[i]]
        neighbors = sample(
            1:(initializer.n_agents), initializer.n_neighbours, replace = false)
        agent = Agent(i, theta, effort, [i], neighbors)
        push!(agents, agent)
    end
    firms = Dict(i => Firm(i, [one(diff_type)], [one(diff_type)],
                     2 * rand(a_generator), 2 * rand(b_generator))
    for i in 1:(initializer.n_agents))
    return agents, firms, a_generator, b_generator
end

"""
    compute_current_firm_utility(agent, current_firm, utility_function)

Compute utility for an agent staying in their current firm.

This function calculates the utility U*(fᵢ) that agent i would receive by staying
in their current firm fᵢ. It computes the optimal effort level and resulting utility
based on the firm's current output and size.

# Arguments
- `agent::Agent{T}`: The agent making the decision
- `current_firm::Firm{T}`: The agent's current firm
- `utility_function::UtilityFunction`: Utility function (typically Cobb-Douglas)

# Returns
- `Tuple`: (stay_utility, firm_id, optimal_effort)
"""
function compute_current_firm_utility(
        agent::Agent{T}, current_firm::Firm{T}, utility_function::UtilityFunction) where {T}
    E_tilde_i = ignore_gradient.(compute_group_effort_from_output(
        current_firm.output[1], agent.effort[1], current_firm.a, current_firm.b))
    optimal_effort = compute_optimal_effort(
        agent.theta, E_tilde_i, current_firm.a, current_firm.b)
    stay_utility = utility_function(current_firm.output[1], current_firm.size[1],
        agent.theta, optimal_effort)
    return stay_utility, current_firm.id, optimal_effort
end

"""
    compute_startup_utility(agent, utility_function, new_firm_a, new_firm_b)

Compute utility for an agent starting a new firm.

This function calculates the utility that agent i would receive by founding a new firm
with production parameters (new_firm_a, new_firm_b). The agent would be the sole employee,
so group effort Ẽᵢ = 0.

# Arguments
- `agent::Agent{T}`: The agent considering starting a firm
- `utility_function::UtilityFunction`: Utility function (typically Cobb-Douglas)
- `new_firm_a`: Linear production parameter a for the new firm
- `new_firm_b`: Quadratic production parameter b for the new firm

# Returns
- `Tuple`: (startup_utility, new_firm_id, optimal_effort)
"""
function compute_startup_utility(
        agent::Agent{T}, utility_function::UtilityFunction, new_firm_a, new_firm_b) where {T}
    new_firm_id = rand(1:Int64(1e10))
    new_firm_output = compute_firms_output(agent.effort[1], new_firm_a, new_firm_b)
    optimal_effort = compute_optimal_effort(agent.theta, zero(T), new_firm_a, new_firm_b)
    startup_utility = utility_function(
        new_firm_output, one(T), agent.theta, optimal_effort)
    return startup_utility, new_firm_id, optimal_effort
end

"""
    compute_neighbor_utilities(agent, agents, firms, utility_function)

Compute utilities for switching to neighboring agents' firms.

This function calculates the utility U*(f) that agent i would receive by switching
to each firm f employed by their neighbors. This enables the agent to discover
firms through their social network.

# Arguments
- `agent::Agent{T}`: The agent considering switching firms
- `agents::Vector{Agent{T}}`: All agents in the simulation
- `firms::Dict{Int64,Firm{T}}`: All firms in the simulation
- `utility_function::UtilityFunction`: Utility function (typically Cobb-Douglas)

# Returns
- `Tuple`: (utilities, firm_ids, optimal_efforts) for each neighbor's firm
"""
function compute_neighbor_utilities(
        agent::Agent{T}, agents::Vector{Agent{T}}, firms::Dict{Int64, Firm{T}},
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

"""
    compute_agent_utilities(agents, agent, firms, utility_function, firm_a_generator, firm_b_generator)

Compute utilities for all firm choices available to an agent.

This function implements the core discrete choice mechanism where agent i computes
utilities U*ᵢ(f) for all firms f ∈ ℱᵢ, where ℱᵢ contains:
1. The agent's current firm
2. A potential new firm the agent could found
3. Firms employed by the agent's neighbors

The agent will then choose the firm that maximizes utility via argmax.

# Arguments
- `agents::Vector{Agent{T}}`: All agents in the simulation
- `agent::Agent{T}`: The agent making the decision
- `firms::Dict{Int64,Firm{T}}`: All firms in the simulation
- `utility_function::UtilityFunction`: Utility function (typically Cobb-Douglas)
- `firm_a_generator`: Random generator for new firm's linear parameter
- `firm_b_generator`: Random generator for new firm's quadratic parameter

# Returns
- `Tuple`: (utilities, firm_candidates_ids, optimal_efforts, new_firm_a, new_firm_b)
"""
function compute_agent_utilities(agents::Vector{Agent{T}}, agent::Agent{T},
        firms::Dict{Int64, Firm{T}}, utility_function::UtilityFunction, firm_a_generator, firm_b_generator) where {T}

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

"""
    update_firm_outputs!(firms, agents)

Update all firm outputs based on current agent efforts.

This function aggregates the effort levels of all agents in each firm and
computes the resulting firm output using the quadratic production function
Q = a·E + b·E², where E is the total effort of all employees.

# Arguments
- `firms::Dict{Int64,Firm{T}}`: Dictionary of all firms (modified in-place)
- `agents::Vector{Agent{T}}`: Vector of all agents
"""
function update_firm_outputs!(
        firms::Dict{Int64, Firm{T}}, agents::Vector{Agent{T}}) where {T}
    # Update firm outputs
    group_efforts = DefaultDict{Int64, T}(zero(T))
    for agent in agents
        firm_id = agent.firm_id[1]
        group_efforts[firm_id] += agent.effort[1]
    end
    for key in keys(group_efforts)
        firms[key].output[1] = compute_firms_output(
            group_efforts[key], firms[key].a, firms[key].b)
    end
end

"""
    step!(agents, firms, utility_function, firm_a_generator, firm_b_generator, activation_rate, delta_t)

Perform one simulation step of the differentiable Axtell Firms model.

This function implements the core discrete firm selection mechanism with surrogate gradients.
Each agent computes utilities for available firms and chooses via differentiable argmax:

During the primal pass: f*ᵢ = argmaxᶠᵨ ∈ ℱᵢ U*ᵢ(f)
During the tangent pass: f̃ᵢ = softmax(U*ᵢ) = exp(U*ᵢ(f)) / ∑ᶠ' ∈ ℱᵢ exp(U*ᵢ(f'))

# Arguments
- `agents::Vector{Agent{T}}`: All agents (modified in-place)
- `firms::Dict{Int64,Firm{T}}`: All firms (modified in-place)
- `utility_function::UtilityFunction`: Utility function for agents
- `firm_a_generator`: Random generator for new firm linear parameters
- `firm_b_generator`: Random generator for new firm quadratic parameters
- `activation_rate`: Rate parameter for agent activation probability
- `delta_t`: Time step size
"""
function step!(agents::Vector{Agent{T}}, firms::Dict{Int64, Firm{T}},
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
                firms[new_firm_id] = Firm(
                    new_firm_id, [zero(T)], [zero(T)], new_firm_a, new_firm_b)
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

"""
    abm_run(params::AxtellFirmsParams)

Run the complete Axtell Firms agent-based model simulation.

This function orchestrates the entire simulation, including agent and firm initialization,
and iterative time steps where agents make differentiable firm selection decisions.
The model captures the emergence of firm size distributions through decentralized
agent interactions and utility maximization.

# Arguments
- `params::AxtellFirmsParams`: Simulation parameters including initializer and dynamics

# Returns
- `Matrix`: Time series data with columns [effort, firm_size, firm_output] over time
"""
function abm_run(params::AxtellFirmsParams)
    utility_function = CobbDouglasUtility()
    diff_type = promote_type(
        get_type(params.agent_initializer), get_type(utility_function))
    agents, firms, firm_a_generator, firm_b_generator = initialize(
        params.agent_initializer, diff_type)
    update_firm_outputs!(firms, agents)
    mean_effort_by_timestep = [mean([agent.effort[1] for agent in agents])]
    mean_firm_output_by_timestep = [mean([firm.output[1]
                                          for firm in values(firms) if firm.size[1] > 0])]
    mean_firm_size_by_timestep = [mean([firm.size[1]
                                        for firm in values(firms) if firm.size[1] > 0])]
    for t in 2:(params.n_steps)
        step!(agents, firms, utility_function, firm_a_generator,
            firm_b_generator, params.activation_rate, params.delta_t)
        push!(mean_effort_by_timestep, mean([agent.effort[1] for agent in agents]))
        push!(mean_firm_output_by_timestep,
            mean([firm.output[1] for firm in values(firms) if firm.size[1] > 0]))
        push!(mean_firm_size_by_timestep,
            mean([firm.size[1] for firm in values(firms) if firm.size[1] > 0]))
    end
    return hcat(
        mean_effort_by_timestep, mean_firm_size_by_timestep, mean_firm_output_by_timestep)'
end

export run_axtell_firms_model, differentiable_histogram

# ... (keep existing utility functions like differentiable_histogram, stable_softmax, etc.)

struct Agent{T}
    id::Int64
    theta::T
    endowment::T
    effort::Vector{T}  # Now a single-element vector
    firm_id::Vector{T}  # Now a single-element vector
    neighbors::Vector{Int64}
end

struct Firm{T}
    id::Int64
    output::Vector{T}  # Now a single-element vector
    size::Vector{T}  # Now a single-element vector
end

function initialize(n_agents, diff_type; max_firms = 100_000)
    agents = Vector{Agent{diff_type}}()
    firms = Dict{Int64, Firm{diff_type}}()
    
    for i in 1:n_agents
        theta = rand(Uniform(0, 1))
        endowment = one(diff_type)
        effort = [convert(diff_type, rand(Uniform(0, 1)))]
        n_neighbors = rand(2:6)
        neighbors = sample(1:n_agents, n_neighbors, replace = false)
        neighbors = setdiff(neighbors, i)
        
        push!(agents, Agent(i, theta, endowment, effort, [i], neighbors))
        firms[i] = Firm(i, [zero(diff_type)], [one(diff_type)])
    end
    
    return agents, firms, n_agents
end

function compute_agent_utilities(agent::Agent{T}, firms::Dict{Int, Firm{T}}, a::T, b::T) where {T}
    utilities = T[]
    candidates = Int[]
    optimal_efforts = T[]
    
    # Current firm
    current_firm = firms[agent.firm_id[1]]
    E_tilde_i = compute_group_effort_from_output(current_firm.output[1], agent.effort[1], a, b)
    optimal_effort = compute_optimal_effort(agent.theta, agent.endowment, E_tilde_i, a, b)
    stay_utility = compute_utility(current_firm.output[1], current_firm.size[1], agent.theta, agent.endowment, optimal_effort)
    
    push!(utilities, stay_utility)
    push!(candidates, agent.firm_id[1])
    push!(optimal_efforts, optimal_effort)
    
    # New firm
    new_firm_output = compute_firms_output(agent.effort[1], a, b)
    optimal_effort = compute_optimal_effort(agent.theta, agent.endowment, zero(T), a, b)
    startup_utility = compute_utility(new_firm_output, one(T), agent.theta, agent.endowment, optimal_effort)
    
    push!(utilities, startup_utility)
    push!(candidates, length(firms) + 1)
    push!(optimal_efforts, optimal_effort)
    
    # Neighbor firms
    for neighbor_id in agent.neighbors
        neighbor_firm = firms[agents[neighbor_id].firm_id[1]]
        E_tilde_i = compute_group_effort_from_output(neighbor_firm.output[1], agent.effort[1], a, b)
        optimal_effort = compute_optimal_effort(agent.theta, agent.endowment, E_tilde_i, a, b)
        switch_utility = compute_utility(neighbor_firm.output[1], neighbor_firm.size[1], agent.theta, agent.endowment, optimal_effort)
        
        push!(utilities, switch_utility)
        push!(candidates, neighbor_firm.id)
        push!(optimal_efforts, optimal_effort)
    end
    
    return utilities, candidates, optimal_efforts
end

function step!(agents::Vector{Agent{T}}, firms::Dict{Int, Firm{T}}, a::T, b::T, activation_rate::T, delta_t::T) where {T}
    activation_probability = 1.0 - exp(-activation_rate * delta_t)
    
    for agent in agents
        if rand() < activation_probability
            utilities, candidates, optimal_efforts = compute_agent_utilities(agent, firms, a, b)
            prob_utilities = stable_softmax(utilities)
            chosen_index = sample(1:length(candidates), Weights(prob_utilities))
            
            new_firm_id = candidates[chosen_index]
            new_effort = optimal_efforts[chosen_index]
            
            # Update firms
            old_firm = firms[agent.firm_id[1]]
            old_firm.size[1] -= 1
            if old_firm.size[1] == 0
                delete!(firms, old_firm.id)
            end
            
            if !haskey(firms, new_firm_id)
                firms[new_firm_id] = Firm(new_firm_id, [zero(T)], [one(T)])
            else
                firms[new_firm_id].size[1] += 1
            end
            
            # Update agent
            agent.effort[1] = new_effort
            agent.firm_id[1] = new_firm_id
        end
    end
    
    # Update firm outputs
    for firm in values(firms)
        group_effort = sum(agent.effort[1] for agent in agents if agent.firm_id[1] == firm.id)
        firm.output[1] = compute_firms_output(group_effort, a, b)
    end
end

function run_axtell_firms_model(n_agents, a, b, activation_rate, delta_t, n_steps)
    agents, firms, n_firms = initialize(n_agents, a)
    firms_sizes_by_time = [Dict(id => firm.size[1] for (id, firm) in firms)]
    
    for t in 1:n_steps
        step!(agents, firms, a, b, activation_rate, delta_t)
        push!(firms_sizes_by_time, Dict(id => firm.size[1] for (id, firm) in firms))
    end
    
    return firms_sizes_by_time
end
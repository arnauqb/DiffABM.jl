import Zygote

export AxtellFirmsParams, CobbDouglasUtility, NeuralNetworkUtility

function initialize(mode::VectorizedAxtellFirms,
        initializer::RandomAxtellAgentInitializer{T}, diff_type) where {T}
    n_agents = initializer.n_agents
    agent_thetas = convert.(diff_type,
        initializer.thetas_bounds[1] .+
        (initializer.thetas_bounds[2] - initializer.thetas_bounds[1]) .* rand(n_agents))
    agent_efforts = convert.(diff_type,
        initializer.initial_efforts_bounds[1] .+
        (initializer.initial_efforts_bounds[2] - initializer.initial_efforts_bounds[1]) .*
        rand(n_agents))
    agent_neighbors = hcat(initializer.neighbours...)
    return agent_thetas, agent_efforts, agent_neighbors
end

function compute_agent_utilities(
        mode::VectorizedAxtellFirms, active_agents_mask, agent_thetas, agent_efforts, agent_neighbors,
        firm_outputs, firm_sizes, agents_to_firms, utility_function::UtilityFunction, a, b)
    n_agents = length(agent_thetas)
    agent_firm_outputs = [firm_outputs[firm_id] for firm_id in agents_to_firms]
    agent_firm_sizes = [firm_sizes[firm_id] for firm_id in agents_to_firms]

    # Current firm utilities and optimal efforts
    effort_without_agent = compute_group_effort_from_output.(
        agent_firm_outputs, agent_efforts, a, b)
    optimal_efforts_stay = compute_optimal_effort.(agent_thetas, effort_without_agent, a, b)
    stay_utilities = utility_function.(agent_firm_outputs, agent_firm_sizes,
        agent_thetas, optimal_efforts_stay)

    # New firm utilities
    new_firm_outputs = compute_firms_output.(agent_efforts, a, b)
    optimal_efforts_new = compute_optimal_effort.(agent_thetas, zeros(n_agents), a, b)
    startup_utilities = active_agents_mask .*
                        utility_function.(
        new_firm_outputs, ones(n_agents), agent_thetas, optimal_efforts_new)

    # Neighbor firm utilities
    n_neighbors = size(agent_neighbors, 1)

    agent_neighbors_firms_ids = agents_to_firms[agent_neighbors]
    agent_neighbors_firms_output = agent_firm_outputs[agent_neighbors]
    agent_neighbors_firms_sizes = agent_firm_sizes[agent_neighbors]

    agent_neighbors_firms_effort_without_agent = compute_group_effort_from_output.(
        agent_neighbors_firms_output',
        agent_efforts,
        a, b
    )
    agent_neighbors_firms_optimal_efforts = compute_optimal_effort.(
        agent_thetas,
        agent_neighbors_firms_effort_without_agent,
        a, b
    )
    agent_neighbors_firms_utilities = utility_function.(
        agent_neighbors_firms_output',
        agent_neighbors_firms_sizes',
        agent_thetas,
        agent_neighbors_firms_optimal_efforts
    ) .* active_agents_mask

    all_utilities = hcat(
        stay_utilities,
        startup_utilities,
        agent_neighbors_firms_utilities
    )
    @assert size(all_utilities) == (n_agents, 2 + n_neighbors)
    all_efforts = hcat(
        optimal_efforts_stay,
        optimal_efforts_new,
        agent_neighbors_firms_optimal_efforts
    )
    @assert size(all_efforts) == (n_agents, 2 + n_neighbors)
    max_firm_ids = maximum(values(agents_to_firms))
    all_candidate_ids = vcat(
        reshape(agents_to_firms, 1, :),
        reshape(collect((max_firm_ids + 1):(max_firm_ids + n_agents)), 1, :),
        agent_neighbors_firms_ids
    )'
    @assert size(all_candidate_ids) == (n_agents, 2 + n_neighbors)
    return all_candidate_ids, all_utilities, all_efforts
end

function step(
        mode::VectorizedAxtellFirms, agent_thetas, agent_efforts, agent_neighbors,
        firm_outputs, firm_sizes, agents_to_firms, utility_function::UtilityFunction,
        a, b, activation_rate, delta_t)
    n_agents = length(agent_thetas)
    activation_probability = 1.0 - exp(-activation_rate * delta_t)
    active_agents_mask = rand(n_agents) .< activation_probability

    candidate_firm_ids, candidate_utilities, candidate_efforts = compute_agent_utilities(
        mode, active_agents_mask, agent_thetas, agent_efforts, agent_neighbors,
        firm_outputs, firm_sizes, agents_to_firms, utility_function, a, b)

    # get one-hot indices for the chosen candidates
    chosen_indices_onehot = reduce(
        hcat, differentiable_argmax.(eachrow(candidate_utilities)))'
    # substract one from old firms
    for firm_id in agents_to_firms
        firm_sizes[firm_id] -= 1
    end
    # update agent firm ids
    new_agents_to_firms = sum(candidate_firm_ids .* chosen_indices_onehot, dims = 2)[:]
    # update firm sizes
    for i in 1:n_agents
        candidate_firms = candidate_firm_ids[i, :]
        for j in axes(candidate_firms, 1)
            firm_id = candidate_firms[j]
            if firm_id in keys(firm_sizes)
                firm_sizes[firm_id] += chosen_indices_onehot[i, j]
            else
                firm_sizes[firm_id] = chosen_indices_onehot[i, j]
            end
        end
    end
    # update agent efforts
    new_agent_efforts = map(1:n_agents) do i
        if active_agents_mask[i]
            sum(candidate_efforts[i] .* chosen_indices_onehot[i])
        else
            agent_efforts[i]
        end
    end
    # update firm outputs
    new_group_efforts = Dict{Int64, Float64}()
    for agent_id in 1:length(agent_thetas)
        firm_id = new_agents_to_firms[agent_id]
        if haskey(new_group_efforts, firm_id)
            new_group_efforts[firm_id] += new_agent_efforts[agent_id]
        else
            new_group_efforts[firm_id] = new_agent_efforts[agent_id]
        end
    end
    new_firm_outputs = Dict(key => compute_firms_output(new_group_efforts[key], a, b) for key in keys(new_group_efforts))
    return new_agents_to_firms, new_agent_efforts, new_firm_outputs
end

function abm_run(params::AxtellFirmsParams{T, M}) where {T, M <: VectorizedAxtellFirms}
    diff_type = promote_type(
        get_type(params.agent_initializer), typeof(params.a[1]), typeof(params.b[1]), get_type(params.utility_function))
    # check that all agents have the same number of neighbors
    agent_thetas, agent_efforts, agent_neighbors = initialize(
        params.mode, params.agent_initializer, diff_type)
    n_agents = length(agent_thetas)
    firm_sizes = Dict(i => 1 for i in 1:(n_agents))
    firm_outputs = Dict(i => 0.0 for i in 1:(n_agents))
    agents_to_firms = collect(1:(n_agents))

    mean_effort_by_timestep = [mean(agent_efforts)]
    mean_firm_output_by_timestep = [mean(values(firm_outputs))]
    mean_firm_size_by_timestep = [mean(values(firm_sizes))]

    for t in 2:(params.n_steps)
        agents_to_firms, agent_efforts, firm_outputs = step(
            VectorizedAxtellFirms(), agent_thetas, agent_efforts, agent_neighbors,
            firm_outputs, firm_sizes, agents_to_firms, params.utility_function, params.a[1], params.b[1],
            params.activation_rate, params.delta_t)

        push!(mean_effort_by_timestep, mean(agent_efforts))
        push!(mean_firm_output_by_timestep, mean([v for v in values(firm_outputs) if v > 0]))
        push!(mean_firm_size_by_timestep, mean([v for v in values(firm_sizes) if v > 0]))
    end

    return hcat(
        mean_effort_by_timestep, mean_firm_size_by_timestep, mean_firm_output_by_timestep)'
end
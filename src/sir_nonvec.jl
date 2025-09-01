export SIRNonVecParams

@kwdef struct SIRNonVecParams{T,S}
    neighbours::Vector{Vector{Int}}
    use_policies::Bool
    model_params::T
    n_agents::Int64
    delta_t::Float64
    n_timesteps::Int64
    discrete_sampler::S
end

@functor SIRNonVecParams (model_params,)

function initialize(sampler, n_agents, initial_infected::T) where {T}
    prob_infected = initial_infected * ones(n_agents)
    is_infected = sample_bernoulli(sampler, prob_infected)
    S = (one(T) .- is_infected)
    I = one(T) .- S
    R = zeros(T, n_agents)
    return S, I, R, I, zeros(T, length(S))
end

function is_active(time, start_time, end_time)
    hard = time >= start_time && time < end_time
    smoothing = GaussianSmoothing(1.0)
    soft = smoothing(time - start_time) * smoothing(end_time - time)
    return hard + (soft - ignore_gradient.(soft))
end

function sample_quarantine(sampler, n_agents, I, time, start_time, end_time, p_quarantine)
    active = is_active(time, start_time, end_time)
    quarantine_probs = ones(n_agents) .* p_quarantine
    does_quarantine = I .* sample_bernoulli(sampler, quarantine_probs)
    return does_quarantine .* active
end

function apply_social_distancing(beta, time, start_time, end_time, alpha)
    active = is_active(time, start_time, end_time)
    return beta * alpha * active + beta * (1.0 - active)
end

function sir_step(model_params, params, S, I, R, time)
    # read parameters
    beta = model_params.beta[1]
    gamma = model_params.gamma[1]
    quarantine_start_time = model_params.quarantine_start_time[1]
    quarantine_end_time = model_params.quarantine_end_time[1]
    p_quarantine = model_params.p_quarantine[1]
    social_distancing_start_time = model_params.social_distancing_start_time[1]
    social_distancing_end_time = model_params.social_distancing_end_time[1]
    alpha = model_params.alpha[1]
    # 

    delta_t = params.delta_t
    n_agents = params.n_agents
    if params.use_policies
        does_quarantine = sample_quarantine(
            params.discrete_sampler, n_agents, I, time, quarantine_start_time,
            quarantine_end_time, p_quarantine)
        beta_effective = apply_social_distancing(
            beta, time, social_distancing_start_time, social_distancing_end_time, alpha)
    else
        does_quarantine = zero(I)
        beta_effective = beta
    end
    delta_I = zero(I)
    delta_R = zero(R)
    prob_infect_vector = zero(I)
    # Pre-compute recovery probability since it's the same for all agents
    prob_recover_vector = (1.0 - exp(-gamma * delta_t)) * ones(n_agents)

    # Pre-compute active status for all agents
    active_status = 1.0 .- does_quarantine

    I_active = I .* active_status

    @inbounds for i in 1:n_agents
        # Get all neighbors at once
        neighbors = params.neighbours[i]

        # Calculate active neighbors and infection rate in one pass
        n_active_neighbours = 0.0
        infection_rate = 0.0
        @inbounds for j in neighbors
            n_active_neighbours += active_status[j]
            infection_rate += I_active[j]
        end
        n_active_neighbours = max(n_active_neighbours, 1.0)

        # Calculate infection probability
        prob_infect = 1.0 - exp(-beta_effective * infection_rate / n_active_neighbours * delta_t)
        prob_infect_vector[i] = prob_infect
    end
    delta_I = S .* sample_bernoulli(params.discrete_sampler, prob_infect_vector)
    delta_R = I .* sample_bernoulli(params.discrete_sampler, prob_recover_vector)
    S = S - delta_I
    I = I + delta_I - delta_R
    R = R + delta_R
    return S, I, R, delta_I, delta_R
end

function abm_run(params::SIRNonVecParams)
    sampler = params.discrete_sampler
    model_params = params.model_params
    n_agents = length(params.neighbours)
    S, I, R, delta_I, delta_R = initialize(sampler, n_agents, model_params.initial_infected[1])

    delta_I_ts = [sum(delta_I)]
    delta_R_ts = [sum(delta_R)]

    for i in 2:(params.n_timesteps)
        time = i * params.delta_t
        S, I, R, delta_I, delta_R = sir_step(model_params, params, S, I, R, time)
        push!(delta_I_ts, sum(delta_I))
        push!(delta_R_ts, sum(delta_R))
    end
    return hcat(delta_I_ts, delta_R_ts)' ./ params.n_agents
end

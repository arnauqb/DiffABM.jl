export SIRNonVecParams

"""
    SIRNonVecParams{T, S}

Parameters for a network-based SIR model with differentiable discrete state transitions.

This model addresses key differentiability challenges in epidemiological simulation:
discrete Bernoulli sampling for infections/recoveries and discrete policy timing controls.
Surrogate gradients enable optimization over both epidemiological parameters (β, γ) 
and policy intervention parameters (timing, compliance rates).

# Fields
- `neighbours::Vector{Vector{Int}}`: Contact network structure
- `use_policies::Bool`: Enable quarantine and social distancing interventions
- `model_params::T`: Epidemiological and policy parameters
- `n_agents::Int64`: Population size
- `delta_t::Float64`: Time step size
- `n_timesteps::Int64`: Simulation duration
- `discrete_sampler::S`: Differentiable sampler for Bernoulli trials
"""
@kwdef struct SIRNonVecParams{T, S}
    neighbours::Vector{Vector{Int}}
    use_policies::Bool
    model_params::T
    n_agents::Int64
    delta_t::Float64
    n_timesteps::Int64
    discrete_sampler::S
end

@functor SIRNonVecParams (model_params,)

"""
    initialize(sampler, n_agents, initial_infected)

Set up initial SIR compartments with differentiable Bernoulli sampling.

Uses differentiable discrete sampling to randomly assign initial infections,
enabling gradient flow through the initial condition parameter I₀.

# Arguments
- `sampler`: Differentiable Bernoulli sampler
- `n_agents`: Population size
- `initial_infected::T`: Initial infection probability I₀

# Returns
- `Tuple`: (S, I, R, delta_I, delta_R) - compartment vectors and initial transitions
"""
function initialize(sampler, n_agents, initial_infected::T) where {T}
    prob_infected = initial_infected * ones(n_agents)
    is_infected = sample_bernoulli(sampler, prob_infected)
    S = (one(T) .- is_infected)
    I = one(T) .- S
    R = zeros(T, n_agents)
    return S, I, R, I, zeros(T, length(S))
end

"""
    is_active(time, start_time, end_time)

Differentiable gate function for policy timing controls.

Replaces the discrete indicator function 𝟙{t_start ≤ t ≤ t_end} with a smooth surrogate
using Gaussian CDF approximation. This enables gradient computation through policy
timing parameters while preserving discrete behavior during forward simulation.

# Arguments
- `time`: Current time
- `start_time`: Policy activation time
- `end_time`: Policy deactivation time

# Returns
- Policy activation strength (discrete: 0 or 1, smooth during backprop)
"""
function is_active(time, start_time, end_time)
    hard = time >= start_time && time < end_time
    smoothing = GaussianSmoothing(1.0)
    soft = smoothing(time - start_time) * smoothing(end_time - time)
    return hard + (soft - ignore_gradient.(soft))
end

"""
    sample_quarantine(sampler, n_agents, I, time, start_time, end_time, p_quarantine)

Sample quarantine compliance with differentiable Bernoulli trials.

Implements the discrete sampling Q_i(t) ~ Bern(p_Q) for infected agents during
active quarantine periods. Uses surrogate gradients to enable differentiation
through the compliance probability p_Q and timing parameters.

# Arguments
- `sampler`: Differentiable Bernoulli sampler
- `n_agents`: Population size  
- `I`: Current infection status
- `time`: Current time
- `start_time`: Quarantine start Q_start
- `end_time`: Quarantine end Q_end
- `p_quarantine`: Compliance probability p_Q

# Returns
- Quarantine status vector
"""
function sample_quarantine(sampler, n_agents, I, time, start_time, end_time, p_quarantine)
    active = is_active(time, start_time, end_time)
    quarantine_probs = ones(n_agents) .* p_quarantine
    does_quarantine = I .* sample_bernoulli(sampler, quarantine_probs)
    return does_quarantine .* active
end

"""
    apply_social_distancing(beta, time, start_time, end_time, alpha)

Apply time-varying transmission reduction during social distancing periods.

Reduces the baseline transmission rate β to β·α_D during the interval 
[D_start, D_end]. The policy timing uses differentiable gate functions 
to enable gradient flow through the intervention parameters.

# Arguments
- `beta`: Baseline transmission rate β
- `time`: Current time
- `start_time`: Social distancing start D_start  
- `end_time`: Social distancing end D_end
- `alpha`: Transmission reduction factor α_D ∈ [0,1]

# Returns
- Effective transmission rate β_eff
"""
function apply_social_distancing(beta, time, start_time, end_time, alpha)
    active = is_active(time, start_time, end_time)
    return beta * alpha * active + beta * (1.0 - active)
end

"""
    sir_step(model_params, params, S, I, R, time)

Execute one time step of network SIR dynamics with differentiable discrete transitions.

Implements the core discrete state transitions through differentiable Bernoulli sampling:
- Infection: I_i(t) ~ Bern(1 - exp(-λ_i(t)Δt)) where λ_i depends on infected neighbors
- Recovery: R_i(t) ~ Bern(1 - exp(-γΔt))
- Quarantine: Q_i(t) ~ Bern(p_Q) during policy periods

Policy interventions modify transmission through differentiable gate functions
that enable gradient flow through timing and intensity parameters.

# Arguments
- `model_params`: Epidemiological (β, γ) and policy parameters
- `params`: Network structure and differentiable samplers
- `S, I, R`: Current compartment status vectors
- `time`: Current simulation time

# Returns
- Updated compartments and transition counts
"""
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
        prob_infect = 1.0 -
                      exp(-beta_effective * infection_rate / n_active_neighbours * delta_t)
        prob_infect_vector[i] = prob_infect
    end
    delta_I = S .* sample_bernoulli(params.discrete_sampler, prob_infect_vector)
    delta_R = I .* sample_bernoulli(params.discrete_sampler, prob_recover_vector)
    S = S - delta_I
    I = I + delta_I - delta_R
    R = R + delta_R
    return S, I, R, delta_I, delta_R
end

"""
    abm_run(params::SIRNonVecParams)

Run differentiable SIR simulation enabling gradient-based parameter optimization.

Combines differentiable discrete sampling for state transitions with smooth surrogate
functions for policy timing controls. This enables gradient computation through all
nine model parameters: epidemiological parameters (β, γ, I₀) and policy intervention
parameters (Q_start, Q_end, p_Q, D_start, D_end, α_D).

# Arguments
- `params::SIRNonVecParams`: Model configuration and parameters

# Returns
- Time series matrix [new_infections, new_recoveries] normalized by population size
"""
function abm_run(params::SIRNonVecParams)
    sampler = params.discrete_sampler
    model_params = params.model_params
    n_agents = length(params.neighbours)
    S, I, R, delta_I, delta_R = initialize(
        sampler, n_agents, model_params.initial_infected[1])

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

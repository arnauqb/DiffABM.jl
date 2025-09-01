using Graphs
using GraphNeuralNetworks
using Distributions
using StatsBase
using Random
using Flux

export SimpleSIRParams, run_simple_sir, SocialDistancing, Quarantine, Policies

# Policy types
struct SocialDistancing{T,V}
    start_time::Vector{V}
    end_time::Vector{V}
    alpha::Vector{T}  # reduction in transmission
end
SocialDistancing() = SocialDistancing(Float64[0.0], Float64[Inf], Float64[1.0])
@functor SocialDistancing (start_time, end_time, alpha)

struct Quarantine{T,V}
    start_time::Vector{V}
    end_time::Vector{V}
    p::Vector{T}  # probability of quarantine
end
Quarantine() = Quarantine(Float64[0.0], Float64[Inf], Float64[0.0])
@functor Quarantine (start_time, end_time, p)

struct Policies{SD,Q}
    social_distancing::SD
    quarantine::Q
end
@functor Policies (social_distancing, quarantine)

# Helper function for policy activation
function is_active(time, start_time, end_time)
    hard = time >= start_time && time < end_time
    smoothing = GaussianSmoothing(0.1)
    soft = smoothing(time - start_time) * smoothing(end_time - time)
    return hard + (soft - ignore_gradient.(soft))
end

# Policy application functions
function (p::SocialDistancing)(x, time)
    mask = is_active(time, p.start_time[1], p.end_time[1])
    return @. x * (mask * p.alpha[1] + (1.0 - mask))
end

function (p::Quarantine)(sampler, time, ones_aux)
    active = is_active(time, p.start_time[1], p.end_time[1])
    quarantine_probs = ones_aux .* p.p[1]
    does_quarantine = sample_bernoulli(sampler, quarantine_probs)
    return does_quarantine .* active
end

@kwdef struct SimpleSIRParams{T,S,P}
    graph::GNNGraph
    initial_infected::Vector{T}  # probability of initial infection
    beta::Vector{T}             # infection rate
    gamma::Vector{T}            # recovery rate
    delta_t::Float64            # time step
    n_timesteps::Int64
    discrete_sampler::S         # for sampling discrete events
    policies::P                 # policies for intervention
end
@functor SimpleSIRParams (initial_infected, beta, gamma, policies)

function initialize(sampler, initial_infected, ones_aux::Vector{T}) where {T}
    prob_infected = initial_infected * ones_aux
    is_infected = sample_bernoulli(sampler, prob_infected)
    S = one(T) .- is_infected
    I = one(T) .- S
    R = one(T) .- ones_aux
    return S, I, R, I, one(T) .- ones_aux
end

function is_complete(graph)
    return ne(graph) == nv(graph) * (nv(graph) - 1)
end

function propagate_infection(
    graph, policies, beta::T, transmission, does_quarantine, time, ones_aux) where {T}
    # check if graph is a complete graph
    # Apply social distancing policies
    beta = policies.social_distancing(beta, time)
    n_non_quarantined_neighbours = propagate(
        (xi, xj, e) -> xj, graph, +, xj=ones_aux, xi=does_quarantine)
    n_non_quarantined_neighbours = max.(one(T), n_non_quarantined_neighbours)
    #if is_complete(graph)
    #    return beta * sum(transmission) ./ n_non_quarantined_neighbours
    #end
    # Simple message passing between agents
    cumulative_trans = propagate(
        (xi, xj, e) -> xi .* xj, graph, +, xi=ones_aux, xj=transmission)
    return beta .* cumulative_trans ./ n_non_quarantined_neighbours
end

function compute_transmission(
    graph, sampler, policies, beta::T, I, time, delta_t, ones_aux) where {T}

    # Apply quarantine policies
    does_quarantine = policies.quarantine(sampler, time, ones_aux)

    # Set transmission to zero for quarantined agents
    transmission = I .* (one(T) .- does_quarantine)

    # Compute infection spread
    transmission = propagate_infection(
        graph, policies, beta, transmission, does_quarantine, time, ones_aux)
    transmission = @. one(T) - exp(-transmission * delta_t)
    return clamp.(transmission, zero(T), one(T))
end

function compute_recovery(gamma::T, delta_t, ones_aux) where {T}
    recovery = (one(T) - exp(-gamma * delta_t)) .* ones_aux
    return clamp.(recovery, zero(T), one(T))
end

function sir_step(graph, S, I, R, beta::T, gamma, time, delta_t,
    sampler, policies, ones_aux) where {T}
    transmission = compute_transmission(
        graph, sampler, policies, beta[1], I, time, delta_t, ones_aux)

    delta_I = S .* sample_bernoulli(sampler, transmission)
    recovery = compute_recovery(gamma[1], delta_t, ones_aux)
    delta_R = I .* sample_bernoulli(sampler, recovery)

    S = S - delta_I
    I = I + delta_I - delta_R
    R = R + delta_R

    return (S, I, R, delta_I, delta_R)
end

function abm_run(params::SimpleSIRParams)
    T = promote_type(eltype(params.initial_infected), eltype(params.gamma),
        eltype(params.beta))

    n_agents = nv(params.graph)
    ones_aux = ones(T, n_agents)
    x = initialize(params.discrete_sampler, params.initial_infected[1], ones_aux)

    delta_I_ts = [sum(x[4])]
    delta_R_ts = [sum(x[5])]

    for i in 2:(params.n_timesteps)
        time = i * params.delta_t
        x = sir_step(
            params.graph, x[1], x[2], x[3],
            params.beta, params.gamma, time, params.delta_t,
            params.discrete_sampler, params.policies, ones_aux)

        delta_I_ts = vcat(delta_I_ts, sum(x[4]))
        delta_R_ts = vcat(delta_R_ts, sum(x[5]))
    end

    return hcat(delta_I_ts, delta_R_ts)' ./ n_agents
end

include("utils.jl")
include("mp.jl")
include("infection.jl")
include("policies.jl")
include("june_graph_loader.jl")

export run_sir, SIRParams

struct SIRParams{G, TI, TB, TG, S, I}
    graph::G
    initial_infected::Vector{TI}
    venue_betas::Vector{TB}
    venues::Vector{Symbol}
    gamma::Vector{TG}
    delta_t::Float64
    n_timesteps::Int64
    discrete_sampler::S
    infection_type::I
    policies::Policies
end
@functor SIRParams (initial_infected, venue_betas, gamma,)
get_betas_by_venue(p::SIRParams) = Dict(zip(p.venues, p.venue_betas))

struct Results{T}
    S_ts::T
    I_ts::T
    R_ts::T
    delta_I_ts::T
    delta_R_ts::T
end

function initialize(sampler, n_agents, initial_infected, T)
    prob_infected = initial_infected * ones(T, n_agents)
    is_infected = sample_bernoulli(sampler, prob_infected)
    S = (1.0 .- is_infected)
    I = 1.0 .- S
    R = zeros(T, n_agents)
    initial_infection_times = zeros(T, n_agents)
    return S, I, R, I, zeros(T, length(S)), initial_infection_times
end

get_edge_type_for_venue(venue) = (:agent, :attends, Symbol(venue))

function propagate_infection(
        graph, policies, venues, betas::Vector{T}, transmission, time) where {T}
    ret = zeros(T, graph.num_nodes[:agent])
    for (venue, base_beta) in zip(venues, betas)
        beta = policies.social_distancing(base_beta, time, venue)
        edge_type = get_edge_type_for_venue(venue)
        ret = ret + propagate_inf(graph, edge_type, beta, transmission)
    end
    return ret
end

function compute_transmission(
        graph, policies, venues, betas, S, I,
        infection_times, infection_type, time, delta_t)
    transmission = @. I * infection_type(time - infection_times)
    transmission = policies.quarantine(sampler, transmission, time)
    transmission = propagate_infection(graph, policies, venues, betas, transmission, time)
    transmission = @. S * (1.0 - exp(-transmission * delta_t))
    transmission = clamp.(transmission, 0.0, 1.0)
    return transmission
end

function compute_transmission(
        graph, policies, venues, betas::Vector{T}, S::Vector{T}, I::Vector{T},
        infection_times::Vector{T}, infection_type, time, delta_t) where {T <:
                                                                          StochasticAD.StochasticTriple}
    return StochasticAD.propagate(
        (betas, S, I, infection_times) -> compute_transmission(
            graph, policies, venues, betas, S, I,
            infection_times, infection_type, time, delta_t),
        betas, S, I, infection_times, keep_deltas = Val(true), provided_st_rep = betas[1])
end

function compute_recovery(I, gamma, delta_t)
    recovery = I .* (1.0 - exp(-gamma * delta_t))
    recovery = clamp.(recovery, 0.0, 1.0)
    return recovery
end

function compute_recovery(I::Vector{T}, gamma::T,
        delta_t) where {T <: StochasticAD.StochasticTriple}
    return StochasticAD.propagate(
        (I, gamma) -> compute_recovery(I, gamma, delta_t),
        I, gamma, keep_deltas = Val(true), provided_st_rep = gamma)
end

function sir_step(graph, S, I, R, infection_times, venues, betas, gamma, time, delta_t,
        sampler, infection_type, policies)
    transmission = compute_transmission(
        graph, policies, venues, betas, S, I,
        infection_times, infection_type, time, delta_t)
    delta_I = sample_bernoulli(sampler, transmission)
    recovery = compute_recovery(I, gamma, delta_t)
    delta_R = sample_bernoulli(sampler, recovery)
    infection_times = infection_times .+ delta_I .* time
    S = S - delta_I
    I = I + delta_I - delta_R
    R = R + delta_R
    x = (S, I, R, delta_I, delta_R, infection_times)
    return x
end

function abm_step(params::SIRParams, x, t)
    x = sir_step(
        params.graph, x[1], x[2], x[3], x[4], params.venues, params.venue_betas,
        params.gamma[1], t, params.delta_t, params.discrete_sampler, params.infection_type, params.policies)
    return x
end

function abm_run(params::SIRParams)
    T = promote_type(eltype(params.initial_infected), eltype(params.gamma),
        eltype(params.venue_betas))
    x = initialize(
        params.discrete_sampler, params.graph.num_nodes[:agent], params.initial_infected[1], T)
    delta_I_ts = [sum(x[4])]
    for t in 2:(params.n_timesteps)
        x = abm_step(params, x, t)
        delta_I_ts = vcat(delta_I_ts, [sum(x[4])])
    end
    return delta_I_ts
end

#function run_sir(graph, initial_infected, betas_by_venue, gamma, delta_t,
#    n_timesteps, sampler, infection_type, policies)
#    T = promote_type(eltype(initial_infected), eltype(gamma), eltype(values(betas_by_venue)))
#    n_agents = graph.num_nodes[:agent]
#    venues = collect(keys(betas_by_venue))
#    betas = collect(values(betas_by_venue))
#    S, I, R, transmission, delta_I, delta_R, infection_times = initialize(
#        sampler, n_agents, initial_infected, T)
#    results = Results(T[sum(S)/n_agents], T[sum(I)/n_agents], T[sum(R)/n_agents],
#        T[sum(I)/n_agents], T[zero(I[1])/n_agents])
#    time = 0.0
#    for _ in 2:n_timesteps
#        step!(
#            graph, S, I, R, transmission, delta_I, delta_R, infection_times, venues, betas, gamma,
#            time, delta_t, sampler, infection_type, policies)
#        # save results
#        push!(results.S_ts, sum(S) / n_agents)
#        push!(results.I_ts, sum(I) / n_agents)
#        push!(results.R_ts, sum(R) / n_agents)
#        push!(results.delta_I_ts, sum(delta_I) / n_agents)
#        push!(results.delta_R_ts, sum(delta_R) / n_agents)
#        time += delta_t
#    end
#    return results
#end

function read_policies(config)
    if "policies" ∉ keys(config)
        return Policies(SocialDistancingPolicies([]), QuarantinePolicies([]))
    end
    pconfig = config["policies"]
    if "social_distancing" ∉ keys(pconfig)
        sd = SocialDistancingPolicies()
    else
        sdconfig = pconfig["social_distancing"]
        sd = []
        for (_, value) in sdconfig
            alphas_by_venue = Dict(Symbol(venue) => alpha
            for (venue, alpha) in value["alphas_by_venue"])
            push!(sd,
                SocialDistancing(
                    value["start_time"], value["end_time"], alphas_by_venue))
        end
        sd = SocialDistancingPolicies(sd)
    end
    if "quarantine" ∉ keys(pconfig)
        qp = QuarantinePolicies()
    else
        qconfig = pconfig["quarantine"]
        qp = []
        for (_, value) in qconfig
            push!(qp, Quarantine(value["start_time"], value["end_time"], value["p"]))
        end
        qp = QuarantinePolicies(qp)
    end
    Policies(sd, qp)
end

function SIRParams(graph, config)
    policies = read_policies(config)
    infection_type = getfield(Main, Symbol(config["infection_type"]["type"]))(config["infection_type"]["params"]...)
    sampler = getfield(Main, Symbol(config["discrete_sampler"]["type"]))(config["discrete_sampler"]["params"]...)
    betas_by_venue = Dict(Symbol(venue) => beta for (venue, beta) in config["venues"])
    betas = collect(values(betas_by_venue))
    venues = collect(keys(betas_by_venue))
    return SIRParams(
        graph,
        config["initial_infected"],
        betas,
        venues,
        config["gamma"],
        config["delta_t"],
        config["n_timesteps"],
        sampler,
        infection_type,
        policies
    )
end

function SIRParams(config::Dict)
    graph = jldopen(config["graph_path"], "r")["graph"]
    return SIRParams(graph, config)
end

function run_sir(config::Dict)
    params = SIRParams(config)
    return run_sir(params)
end

function convert_graph(graph, type)
    num_nodes = graph.num_nodes
    eindex = graph.graph
    edata = Dict(
        edge_type => (aux1 = zeros(type, length(eindex[edge_type][1])),
            aux2 = zeros(type, length(eindex[edge_type][1])))
    for edge_type in keys(eindex)
    )
    ndata = Dict{Symbol, NamedTuple}(
        edge_type[3] => (beta = ones(type, num_nodes[edge_type[3]]),)
    for edge_type in keys(eindex)
    )
    ndata[:agent] = (lambda = zeros(type, num_nodes[:agent]),)
    return GNNHeteroGraph(eindex; num_nodes, ndata, edata)
end

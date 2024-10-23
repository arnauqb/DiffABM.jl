export symmetrize_edges, generate_random_world_graph

"""
Given a named tuple of edges, adds the reverse edges to the named tuple.
"""
function symmetrize_edges(edges)
    # turn edges to dict
    edges_dict = Dict()
    for (k, v) in edges
        edges_dict[k] = v
    end
    # add reverse edges
    for (k, v) in edges
        edges_dict[(k[3], k[2], k[1])] = (v[2], v[1])
    end
    # convert back to named tuple
    return (k => v for (k, v) in edges_dict)
end

"""
PDF of Gamma distribution modified to avoid NaNs in backprop.
"""
function gamma_pdf(μ, σ, x)
    α = μ^2 / σ^2
    β = μ / σ^2
    ret = @. β^α / gamma(α) * x^(α - 1) * exp(-β * x)
    return ret
end

function generate_random_world_graph(
        n_agents, venues, number_per_venue, prob_per_venue)
    n_venues = length(venues)
    num_nodes = Dict(:agent => n_agents)
    eindex_dict = Dict()
    edata = Dict()
    for i in 1:n_venues
        n_this_venue = number_per_venue[i]
        prob_attendance_venue = prob_per_venue[i]
        agents_connect = rand(Bernoulli(prob_attendance_venue), n_agents)
        venues_to_connect = rand(1:n_venues, n_agents)
        senders = collect(1:n_agents)[agents_connect]
        receivers = venues_to_connect[agents_connect]
        venue_symbol = venues[i]
        prob_attendance_venue = prob_per_venue[i]
        eindex_dict[(:agent, :attends, venue_symbol)] = (senders, receivers)
        eindex_dict[(venue_symbol, :attends, :agent)] = (receivers, senders)
        num_nodes[venue_symbol] = n_this_venue
    end
    eindex = (k => v for (k, v) in eindex_dict)
    edata = (k => v for (k, v) in edata)
    return GNNHeteroGraph(eindex; num_nodes, edata)
end
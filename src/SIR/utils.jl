export symmetrize_edges

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
        n_agents, venues, fraction_population_per_venue, number_per_venue)
    n_venues = length(venues)
    num_nodes = Dict(:agent => n_agents)
    eindex_dict = Dict()
    for i in 1:n_venues
        n_agents_in_venue = Int(floor(n_agents * fraction_population_per_venue[i]))
        agents_in_venue = randperm(n_agents)[1:n_agents_in_venue]
        n_people_per_venue = number_per_venue[i]
        edges_venue = agents_in_venue, rand(1:n_people_per_venue, n_agents_in_venue)
        venue_symbol = Symbol(venues[i])
        eindex_dict[(:agent, :attends, venue_symbol)] = edges_venue
        eindex_dict[(venue_symbol, :attends, :agent)] = (edges_venue[2], edges_venue[1])
        num_nodes[venue_symbol] = n_people_per_venue
    end
    eindex = (k => v for (k, v) in eindex_dict)
    return GNNHeteroGraph(eindex; num_nodes)
end

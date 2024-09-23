function _message(xi, xj, e)
    return xi .* xj
end

function _message_edge_weights(xi, xj, e)
    return xi .* xj .* e
end

function get_number_neighbors(graph, edge_type)
    subgraph = edge_type_subgraph(graph, edge_type)
    n_agents = subgraph.num_nodes[edge_type[1]]
    n_locations = subgraph.num_nodes[edge_type[3]]
    return propagate(_message, subgraph, +, xi = ones(n_locations), xj = ones(n_agents))
end

function propagate_inf(
        graph, edge_type, beta::T, transmission::AbstractVector{Q}) where {T, Q}
    sg = edge_type_subgraph(graph, edge_type)
    betas = beta .* ones(T, graph.num_nodes[edge_type[3]])
    edge_weight = sg.edata[edge_type].e
    #cumulative_trans = propagate(
    #    _message_edge_weights, sg, +, xi = betas, xj = transmission, e = edge_weight)
    cumulative_trans = propagate(
        _message, sg, +, xi = betas, xj = transmission)
    n_neighbors = max.(1.0, get_number_neighbors(sg, edge_type) .- 1.0)
    cumulative_trans = cumulative_trans ./ n_neighbors
    reverse_edge_type = (edge_type[3], edge_type[2], edge_type[1])
    # send back to agents
    sg = edge_type_subgraph(graph, reverse_edge_type)
    cumulative_trans = propagate(
        _message, sg, +, xi = ones(T, length(transmission)), xj = cumulative_trans)
    return cumulative_trans
end
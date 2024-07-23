export read_june_graph

function read_hdf5_as_dict(filename::String)
    ret = Dict()
    c = h5open(filename, "r") do file
        for key in keys(file)
            if !occursin("leisure", key)
                ret[key] = read(file, key)
            end
        end
    end
    return ret
end

function parse_dict_to_gnn(data::Dict, n_people_per_leisure_venue)
    # read number of nodes
    num_nodes = Dict()
    for key in keys(data)
        if occursin("id", key)
            name = join(split(key, "_")[1:end-1], "_")
            num_nodes[Symbol(name)] = length(data[key])
        end
    end
    n_leisure_venues = ceil(Int, num_nodes[:agent] / n_people_per_leisure_venue)
    num_nodes[:leisure] = n_leisure_venues
    # read edges
    edges_dict = Dict()
    for key in keys(data)
        if occursin("edge", key)
            edges = data[key]
            name = join(split(key, "_")[2:end], "_")
            edges_dict[(:agent, :attends, Symbol(name))] = (edges[:, 1] .+ 1, edges[:, 2] .+ 1)
            edges_dict[(Symbol(name), :attends, :agent)] = (edges[:, 2] .+ 1, edges[:, 1] .+ 1)
        end
    end
    eindex = (k => v for (k, v) in edges_dict)
    edges_leisure = randperm(num_nodes[:agent]), rand(1:n_leisure_venues, num_nodes[:agent])
    edges_dict[(:agent, :attends, :leisure)] = edges_leisure
    edges_dict[(:leisure, :attends, :agent)] = (edges_leisure[2], edges_leisure[1])
    edata = Dict(
        edge_type => (aux1=zeros(Real, length(edges_dict[edge_type][1])),
            aux2=zeros(Real, length(edges_dict[edge_type][1])))
        for edge_type in keys(edges_dict)
    )
    ndata = Dict{Symbol, NamedTuple}(
        edge_type[3] => (beta=ones(Real, num_nodes[edge_type[3]]),) for edge_type in keys(edges_dict)
    )
    ndata[:agent] = (lambda=zeros(Real, num_nodes[:agent]),)
    return GNNHeteroGraph(eindex; num_nodes, ndata, edata)
end

function read_june_graph(filename::String; n_people_per_leisure_venue=20)
    data = read_hdf5_as_dict(filename)
    return parse_dict_to_gnn(data, n_people_per_leisure_venue)
end
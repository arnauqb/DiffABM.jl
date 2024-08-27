using DiffABM
using Test

@testset "test SIR" begin
    n_agents = 100
    venues = Symbol.(["household", "company", "school", "leisure"])
    fraction_population_per_venue = [1.0, 0.4, 0.4, 1.0]
    number_per_venue = [1, 2, 3, 4]
    graph = DiffABM.generate_random_world_graph(
        n_agents, venues, fraction_population_per_venue, number_per_venue)
    graph_generator() = graph
    initial_infected = [0.1]
    gamma = [0.05]
    betas = [0.5, 0.2, 0.3, 0.1]
    n_timesteps = 30
    delta_t = 1.0
    infection_type = ConstantInfection()
    policies = Policies()
    for discrete_sampler in [GS(0.1), SAD(), ST(), SM()]
        params = SIRParams(
            graph_generator, initial_infected, betas, venues, gamma, delta_t, n_timesteps,
            discrete_sampler, infection_type, policies)
        delta_I_ts = abm_run(params)
        @test length(delta_I_ts) == n_timesteps
    end
end
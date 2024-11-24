using DiffABM
using PyPlot

##
struct ConstantGraph{T}
    graph::T
end
(cg::ConstantGraph)() = cg.graph

function make_sir(;
        initial_infected,
        betas,
        gamma,
        social_distancing_start_date,
        social_distancing_end_date,
        social_distancing_probs,
        quarantine_start_date,
        quarantine_end_date,
        quarantine_prob
)
    n_agents = 10000
    venues = [:venue]
    graph = generate_complete_graph(n_agents)
    generator = ConstantGraph(graph)
    # format: start date, end date, quarantine prob
    quarantine = Quarantine(
        [quarantine_start_date], [quarantine_end_date], [quarantine_prob])
    quarantine_policies = QuarantinePolicies([quarantine])
    # format: start date, end date, venues, social distancing prob
    social_distancing = SocialDistancing(
        [social_distancing_start_date], [social_distancing_end_date],
        [:venue], social_distancing_probs)
    social_distancing_policies = SocialDistancingPolicies([social_distancing])
    policies = Policies(social_distancing_policies, quarantine_policies)
    initial_infected = [initial_infected]
    betas = betas # beta per each venue
    gamma = [gamma]
    delta_t = 1.0
    n_timesteps = 60
    sir_params = SIRParams(generator, initial_infected, betas, venues, gamma, delta_t,
        n_timesteps, SAD(), ConstantInfection(), policies)
    return sir_params
end
sir_params = make_sir(
    initial_infected = 0.005,
    betas = [0.3],
    gamma = 0.05,
    social_distancing_start_date = 12.0,
    social_distancing_end_date = 40.0,
    social_distancing_probs = [0.5],
    quarantine_start_date = 20.0,
    quarantine_end_date = 30.0,
    quarantine_prob = 0.8
)
x = abm_run(sir_params)

##
fig, ax = plt.subplots()
ax.plot(x[1,:], label = "infected")
ax.plot(x[2,:], label = "recovered")
ax.set_ylabel("proportion of population")
ax.set_xlabel("timestep")
ax.legend()
fig

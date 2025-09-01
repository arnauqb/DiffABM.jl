using Graphs
using Random
using Distributions
using GraphNeuralNetworks
using PyPlot
using DiffABM

##

# Set random seed for reproducibility
Random.seed!(123)

# Create network
n_agents = 10000
graph = GNNGraph(erdos_renyi(n_agents, 0.01))

# Create policies
sd_policy = SocialDistancing(
    [7.0],  # start time
    [25.0],  # end time
    [0.5]    # transmission reduction (30% reduction)
)

q_policy = Quarantine(
    [12.0],  # start time
    [20.0],  # end time
    [0.5]    # quarantine probability (10%)
)

policies = Policies(sd_policy, q_policy)

# Create parameters
params = SimpleSIRParams(
    graph,
    [0.005],          # initial_infected (1% of population)
    [0.3],           # beta (infection rate)
    [0.05],           # gamma (recovery rate)
    1.0,             # delta_t (time step)
    40,             # n_timesteps
    ST(),  # discrete_sampler
    policies
)

# Run simulation
results = run_simple_sir(params)

# Plot results
# Plot results
fig, ax = subplots()
ax.plot(1:params.n_timesteps, results[1, :], label="New Infections")
ax.plot(1:params.n_timesteps, results[2, :], label="New Recoveries")

# Add vertical lines for policy interventions
ax.axvline(x=params.policies.quarantine.start_time[1]-1, label="Quarantine Start", linestyle="--")
ax.axvline(x=params.policies.quarantine.end_time[1]-1, label="Quarantine End", linestyle="--")
ax.axvline(x=params.policies.social_distancing.start_time[1]-1, label="SD Start", linestyle=":")
ax.axvline(x=params.policies.social_distancing.end_time[1]-1, label="SD End", linestyle=":")

ax.set_xlabel("Time Steps")
ax.set_ylabel("Fraction of Population")
ax.set_title("SIR Model with Interventions")
ax.legend()
fig
##

# Save the plot
#savefig(plotsdir("sir_simple_results.png"))

# Print some summary statistics
total_infected = sum(results[1, :])
total_recovered = sum(results[2, :])
peak_infection = maximum(results[1, :])
time_to_peak = findmax(results[1, :])[2]

println("\nSimulation Results:")
println("Total infected: $(round(total_infected * 100, digits=2))% of population")
println("Total recovered: $(round(total_recovered * 100, digits=2))% of population")
println("Peak infection: $(round(peak_infection * 100, digits=2))% of population")
println("Time to peak: $(time_to_peak * params.delta_t) days")

# Run comparison without interventions
no_intervention_policies = Policies(
    SocialDistancing([999.0], [1000.0], [1.0]),  # Never activates
    Quarantine([999.0], [1000.0], [0.0])         # Never activates
)

params_no_intervention = SimpleSIRParams(
    graph,
    [0.01],          # initial_infected
    [0.3],           # beta
    [0.1],           # gamma
    0.1,             # delta_t
    200,             # n_timesteps
    BernoulliSampler(0.5),  # discrete_sampler
    no_intervention_policies
)

results_no_intervention = run_simple_sir(params_no_intervention)

# Plot comparison
fig2, ax2 = subplots()
ax2.plot(1:params.n_timesteps, results[1, :], label="With Interventions")
ax2.plot(1:params.n_timesteps, results_no_intervention[1, :], label="No Interventions")

ax2.set_xlabel("Time Steps")
ax2.set_ylabel("New Infections (Fraction of Population)")
ax2.set_title("Effect of Interventions on Infection Rate")
ax2.legend()

# Save the comparison plot
#savefig(plotsdir("sir_simple_comparison.png"))

# Print intervention effectiveness
peak_with = maximum(results[1, :])
peak_without = maximum(results_no_intervention[1, :])
reduction = (1 - peak_with/peak_without) * 100

println("\nIntervention Effectiveness:")
println("Peak infection reduction: $(round(reduction, digits=2))%")

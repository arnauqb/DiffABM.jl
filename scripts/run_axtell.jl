using DiffABM
using StatsBase
using PyPlot
using Flux
using Random
using Distributions

##
function make_axtell_params(theta_bounds, initial_effort_bounds, a, b)
    n_agents = 1000
    n_timesteps = 100
    update_rate = 0.05
    delta_t = 1.0
    gradient_horizon = 200
    neighbors = [sample(1:n_agents, rand(2:6), replace = false) for _ in 1:n_agents]
    agent_initializer = RandomAxtellAgentInitializer(n_agents, theta_bounds, initial_effort_bounds, neighbors)
    return AxtellFirmsParams(agent_initializer, [a], [b], update_rate, delta_t, n_timesteps, gradient_horizon)
end
abm_params = make_axtell_params([0.1, 0.9], [0.0, 1.0], 1.5, 2.0)
# once we have this object, we can just do
x = abm_run(abm_params)

##
fig, ax = plt.subplots()
ax.plot(x[1,:])
ax.set_title("mean firm output")
ax.set_xlabel("timestep")
fig

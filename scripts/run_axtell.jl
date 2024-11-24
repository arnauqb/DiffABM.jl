using DiffABM
using StatsBase
using PyPlot
using Flux
using Random
using Distributions

##
"""
Here bounds represents the parameters for a beta distribution.
"""
function make_axtell_params(theta_bounds, initial_effort_bounds, a_bounds, b_bounds)
    n_agents = 1000
    n_timesteps = 30
    update_rate = 0.25
    delta_t = 1.0
    gradient_horizon = 10000
    n_neighbors = 4
    agent_initializer = RandomAxtellAgentInitializer(
        n_agents,
        theta_bounds,
        initial_effort_bounds,
        a_bounds,
        b_bounds,
        n_neighbors)
    return AxtellFirmsParams(
        agent_initializer, update_rate, delta_t, n_timesteps, gradient_horizon)
end
theta_bounds = [1.0, 3.0]
initial_effort_bounds = [2.0, 1.0]
a_bounds = [2.0, 5.0]
b_bounds = [5.0, 2.0]
abm_params = make_axtell_params(theta_bounds, initial_effort_bounds, a_bounds, b_bounds)
x = abm_run(abm_params)

##
fig, axs = plt.subplots(1, 3, figsize = (10, 3))
axs[1].plot(x[1,:])
axs[1].set_title("Mean effort per time-step")
axs[2].plot(x[2,:])
axs[2].set_title("Mean firm size per time-step")
axs[3].plot(x[3,:])
axs[3].set_title("Mean firm output per time-step")
for ax in axs
    ax.set_xlabel("timestep")
end
fig

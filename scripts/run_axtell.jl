using DiffABM
using StatsBase
using PyPlot
using ForwardDiff
using Flux
using Random
using Distributions

##
function make_abm_params(a, b)
    n_agents = 1000
    n_timesteps = 100
    neighbors = [sample(1:n_agents, rand(2:6), replace = false) for _ in 1:n_agents]
    agent_initializer = DiffABM.RandomAxtellAgentInitializer(n_agents, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), neighbors)
    return DiffABM.AxtellFirmsParams(agent_initializer, [a], [b], 0.05, 1.0, n_timesteps, 1.0)
end
function run(params)
    Random.seed!(1)
    abm_params = make_abm_params(params...)
    return DiffABM.abm_run(abm_params)
end
jacobian = ForwardDiff.jacobian(run, [0.1, 0.9])
##

params_to_run = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for (i, params) in enumerate(params_to_run)
    outputs = run(params);
    ax[1].plot(outputs[:,1], label="a=$(params[1]), b=$(params[2])")
    ax[2].plot(outputs[:,2], label="a=$(params[1]), b=$(params[2])")
end
ax[1].set_title("Mean Agent Effort")
ax[2].set_title("Mean Firm Output")
ax[1].legend()
ax[2].legend()
fig

##
jacobian = ForwardDiff.jacobian(run, [0.1, 0.9])
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[1].plot(jacobian[1:100, 1])
ax[1].plot(jacobian[1:100, 2])
ax[2].plot(jacobian[101:end, 1])
ax[2].plot(jacobian[101:end, 2])
fig
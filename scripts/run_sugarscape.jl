using DiffABM
using Distributions
using DistributionsAD
using Flux
using StaticArrays
using DelimitedFiles
using PyPlot

##
function make_sugarscape_params(vision_probs, metabolic_rate_probs)
	board_length = 50
	n_agents = 100
	n_timesteps = 30
    board = readdlm("scripts/sugar-map.txt")[:]
	board_initializer = GeneratedBoard(board_length, board)
	positions = [rand(1:board_length, 2) for i in 1:n_agents]
	max_age_distribution = Uniform(60.0, 100.0)
	wealth_distribution = DiscreteUniform(6, 25)
    sugar_regeneration_rate = 1.0
    gradient_horizon = 500
	agent_initializer = RandomAgentInitializer(vision_probs, metabolic_rate_probs, max_age_distribution, wealth_distribution, positions)
	moving_rule = ArgmaxMovingRule(VonNeumannNeighborhood(board_length, length(vision_probs)))
    smoothing = GaussianSmoothing(1.0)
	sugarscape = SugarScapeParams(
		board_initializer, agent_initializer, moving_rule, board_length,
		n_agents, n_timesteps, [sugar_regeneration_rate], gradient_horizon, smoothing)
	return sugarscape
end
sugarscape_params = make_sugarscape_params([0.0, 0.0, 1.0], [0.75, 0.25]);
x = abm_run(sugarscape_params);
# this outputs: board_history, x_history, y_history, wealth_history, alive_history, occupied_history
# note that these have all the agents!

##
fig, ax = plt.subplots()
# plot total wealth history
wealth_history_ts = [sum(x[4][i]) for i in 1:length(x[4])]
ax.plot(wealth_history_ts)
ax.set_title("total wealth")
ax.set_xlabel("timestep")
fig

using DiffABM
using Distributions
using DistributionsAD
using Plots
using Flux
using StaticArrays

##
board_length = 50
n_agents = 250
n_timesteps = 100
peak_positions = [0.1 * board_length, 0.1 * board_length, 0.9 * board_length, 0.9 * board_length]
max_sugar = 5.0
sugar_regeneration_rate = [0.1]
distance = (x, y) -> sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2) / 5
board_initializer = TwoPeakBoard(board_length, peak_positions, [max_sugar], distance)
agent_initializer = RandomAgentInitializer(board_length)
moving_rule = ArgmaxMovingRule(VonNeumannNeighborhood(board_length, 3))
sugarscape = SugarScapeParams(board_initializer, agent_initializer, moving_rule, board_length, n_agents, n_timesteps, sugar_regeneration_rate);

##
board_history, x_history, y_history, alive_history, occupied_history = abm_run(sugarscape);

p = Plots.plot()
alive_ts = [sum(alive_history[i]) for i in 1:n_timesteps]
plot!(alive_ts)

##
function plot_board(board, x, y, alive)
    p = Plots.plot()
    board_x = collect(1:size(board)[1])
    board_y = collect(1:size(board)[2])
    heatmap!(p, board_x, board_y, board', clim = (0, max_sugar))
    # scatter with white color for alive and red for dead
    alive_color = [Bool(alive) ? :white : :black for alive in alive]
    scatter!(p, x, y, c=alive_color, ms=2)
    p
end

plot_board(board_history[1], x_history[1], y_history[1], alive_history[1])

## animate
anim = @animate for i in 1:n_timesteps
    plot_board(board_history[i], x_history[i], y_history[i], alive_history[i])
end
gif(anim, "sugarscape.gif", fps=10)

##
function plot_occupied(occupied)
    p = plot()
    board_x = collect(1:size(occupied)[1])
    board_y = collect(1:size(occupied)[2])
    heatmap!(p, board_x, board_y, occupied', clim = (0, 1))
    p
end

anim = @animate for i in 1:n_timesteps
    plot_occupied(occupied_history[i])
end
gif(anim, "occupied.gif", fps=10)

using BenchmarkTools
using DiffABM
using Distributions
using DistributionsAD
using Flux
using Images
using DelimitedFiles
using PyPlot
using Profile, PProf
using Random
using PyCall
animation = pyimport("matplotlib.animation")

##
function make_sugarscape_params(vision_probs, metabolic_rate_bounds, wealth_bounds)
    board_length = 50
    n_agents = 100
    n_timesteps = 100
    board = readdlm("scripts/sugar-map.txt")[:]
    board = Images.imresize(board, (board_length, board_length))[:]
    board_initializer = GeneratedBoard(board_length, board)
    discrete_sampler = SM()
    neighborhood = VonNeumannNeighborhood()
    sugar_regeneration_rate = 1.0
    gradient_horizon = 10000
    agent_initializer = RandomAgentInitializer(
        board_length,
        vision_distribution_probs = vision_probs,
        metabolic_rate_bounds = metabolic_rate_bounds,
        wealth_bounds = wealth_bounds,
        neighborhood = neighborhood,
        discrete_sampler = discrete_sampler)
    smoothing = GaussianSmoothing(1.0)
    sugarscape = SugarScapeParams(
        board_initializer, agent_initializer, board_length,
        n_agents, n_timesteps, [sugar_regeneration_rate], gradient_horizon, smoothing)
    return sugarscape
end
vision_probs = ones(6) ./ 6 #[0.2, 0.2, 0.2, 0.2, 0.2]
metabolic_rate_bounds = [2.0, 5.0] # define a beta distribution
wealth_bounds = [5.0, 2.0] # define a beta distribution
sugarscape_params = make_sugarscape_params(vision_probs, metabolic_rate_bounds, wealth_bounds);
# this outputs: board_history, x_history, y_history, wealth_history, alive_history, occupied_history
# note that these have all the agents!
##
function summarizer(x)
    n_timesteps = length(x[5])
    n_agents = length(x[5][1])
    alive_per_timestep = [sum(x[5][t]) / n_agents for t in 1:n_timesteps]
    return reshape(alive_per_timestep, 1, :)
end

##
params = make_sugarscape_params(vision_probs, metabolic_rate_bounds, wealth_bounds);
board_history, x_history, y_history, wealth_history, alive_history, occupied_history = abm_run(params);

##
# make a movie with the ts
function run_and_animate(vision_probs, metabolic_rate_bounds, wealth_bounds)
    function update_plot(frame)
        frame = frame + 1
        ax.clear()
        board = board_history[frame]
        #board = occupied_history[frame]
        x = x_history[frame]
        y = y_history[frame]
        alive = alive_history[frame]

        # Use pcolormesh with plasma colormap
        im = ax.pcolormesh(board', cmap = "inferno")#, vmin = 0, vmax = 5.0)
        #fig.colorbar(im, ax=ax)

        # Scatter plot with white color for alive and black for dead
        alive_color = [a == 1.0 ? "white" : "black" for a in alive]
        ax.scatter(x, y, c = alive_color, s = 10, edgecolors = "none")

        ax.set_title("Timestep $frame")
        ax.set_xlim(0, size(board, 1))
        ax.set_ylim(0, size(board, 2))
        return [im]
    end

    #Random.seed!(1234)
    sugarscape_params = make_sugarscape_params(vision_probs, metabolic_rate_bounds, wealth_bounds);
    board_history, x_history, y_history, wealth_history, alive_history, occupied_history = abm_run(sugarscape_params)

    # Create the figure and axis outside the animation function
    fig, ax = plt.subplots()
    board = board_history[1]
    #board = occupied_history[1]
    im = ax.pcolormesh(board', cmap = "inferno")#, vmin = 0, vmax = 1.0)
    fig.colorbar(im, ax = ax)
    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        update_plot,
        frames = sugarscape_params.n_timesteps,
        interval = 100,  # 100 ms between frames
        blit = true,
        repeat = false
    )

    # Save the animation
    anim.save("sugarscape.gif", writer = "pillow", fps = 10)
    plt.close(fig)
end
run_and_animate(vision_probs, metabolic_rate_bounds, wealth_bounds)

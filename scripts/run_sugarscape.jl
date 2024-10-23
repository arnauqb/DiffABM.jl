using DiffABM
using Distributions
using DistributionsAD
using Flux
using DelimitedFiles
using PyPlot
using Random
using PyCall
animation = pyimport("matplotlib.animation")

##
function make_sugarscape_params(vision_probs, metabolic_rate_probs)
    board_length = 50
    n_agents = 1000
    n_timesteps = 50
    board = readdlm("scripts/sugar-map.txt")[:]
    board_initializer = GeneratedBoard(board_length, board)
    #positions = [rand(1:board_length, 2) for i in 1:n_agents]
    max_age_distribution = Uniform(60.0, 100.0)
    wealth_distribution = DiscreteUniform(6, 25)
    discrete_sampler = ST()
    neighborhood = VonNeumannNeighborhood()
    sugar_regeneration_rate = 1.0
    gradient_horizon = 500
    agent_initializer = RandomAgentInitializer(
        board_length; vision_distribution_probs = vision_probs,
        metabolic_rate_probs = metabolic_rate_probs,
        neighborhood = neighborhood, discrete_sampler = discrete_sampler,
        wealth_distribution = wealth_distribution, max_age_distribution = max_age_distribution)
    smoothing = GaussianSmoothing(1.0)
    sugarscape = SugarScapeParams(
        board_initializer, agent_initializer, board_length,
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

##
# make a movie with the ts
function run_and_animate(vision_probs, metabolic_rate_probs)
    function update_plot(frame)
        frame = frame + 1
        ax.clear()
        board = board_history[frame]
        x = x_history[frame]
        y = y_history[frame]
        alive = alive_history[frame]

        # Use pcolormesh with plasma colormap
        im = ax.pcolormesh(board', cmap = "inferno", vmin = 0, vmax = 5.0)
        #fig.colorbar(im, ax=ax)

        # Scatter plot with white color for alive and black for dead
        alive_color = [a == 1.0 ? "white" : "black" for a in alive]
        ax.scatter(x, y, c = alive_color, s = 10, edgecolors = "none")

        ax.set_title("Timestep $frame")
        ax.set_xlim(0, size(board, 1))
        ax.set_ylim(0, size(board, 2))
        return [im]
    end

    Random.seed!(1234)
    sugarscape_params = make_sugarscape_params(vision_probs, metabolic_rate_probs);
    board_history, x_history, y_history, alive_history, occupied_history = abm_run(sugarscape_params)

    # Create the figure and axis outside the animation function
    fig, ax = plt.subplots()
    board = board_history[1]
    im = ax.pcolormesh(board', cmap = "inferno", vmin = 0, vmax = 5.0)
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
run_and_animate([0.0, 0.0, 1.0], [0.75, 0.25])

using DiffABM
using Distributions
using FiniteDifferences
using Flux
using DifferentiationInterface
using Functors
using LinearAlgebra
using Zygote
using PyPlot
using PyCall
using StochasticAD
animation = pyimport("matplotlib.animation")
using Random
include("../test/utils.jl")

##
function run_and_animate(params)
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
    model = make_model(params)
    board_history, x_history, y_history, alive_history, occupied_history = abm_run(model)

    max_sugar = model.board_initializer.max_sugar[1]

    # Create the figure and axis outside the animation function
    fig, ax = plt.subplots()
    board = board_history[1]
    im = ax.pcolormesh(board', cmap = "inferno", vmin = 0, vmax = 5.0)
    fig.colorbar(im, ax = ax)
    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        update_plot,
        frames = model.n_timesteps,
        interval = 100,  # 100 ms between frames
        blit = true,
        repeat = false
    )

    # Save the animation
    anim.save("sugarscape.gif", writer = "pillow", fps = 10)
    plt.close(fig)
end
function make_model(params; gradient_horizon)
    board_length = 50
    n_agents = 200
    n_timesteps = 100
    max_sugar = params[1]
    sugar_regeneration_rate = 0.1
    peak_positions = [
        0.1 * board_length, 0.1 * board_length, 0.9 * board_length, 0.9 * board_length]
    #   vision_probs = zeros(5)
    #   vision_probs[end] = 1.0
    #vision_distribution = Categorical(vision_probs)
    distance_function = (a, b) -> sqrt(sum((a .- b) .^ 2) + 1e-3) / 5
    board_initializer = DiffABM.TwoPeakBoard(
        board_length, peak_positions, [max_sugar], distance_function)
    ages = 50 .* ones(n_agents)
    metabolic_rates = 4 .* ones(n_agents)
    vision = 6 .* ones(n_agents)
    wealth = 20 .* ones(n_agents)
    # distribute agents in a grid uniformly at random
    positions = [(i, j) for i in 1:board_length, j in 1:board_length][randperm(n_agents)]
    #agent_initializer = DiffABM.GeneratedAgentInitializer(vision, metabolic_rates, ages, wealth, positions)
    agent_initializer = DiffABM.RandomAgentInitializer(board_length)
    moving_rule = ArgmaxMovingRule(MooreNeighborhood(board_length, 5))
    sugarscape = SugarScapeParams(
        board_initializer, agent_initializer, moving_rule, board_length,
        n_agents, n_timesteps, [sugar_regeneration_rate], gradient_horizon)
    return sugarscape
end
function run_model(params; gradient_horizon)
    Random.seed!(1234)
    model = make_model(params, gradient_horizon = gradient_horizon)
    board_history, x_history, y_history, wealth_history, alive_history, occupied_history = abm_run(model)
    #return [sum(wealth_history[i]) for i in 1:(model.n_timesteps)]
    #return [sum(board_history[i]) for i in 1:model.n_timesteps]
    return [sum(alive_history[i]) for i in 1:model.n_timesteps]
end
##
params = [10.0]
params_to_try = [5.0, 10, 20]
ts = [run_model(params_to_try[i], gradient_horizon = 101) for i in 1:length(params_to_try)]
fig, ax = plt.subplots()
for i in 1:length(params_to_try)
    ax.plot(ts[i], label = "regen rate = $(params_to_try[i])")
end
ax.set_xlabel("Timestep")
ax.legend()
fig

##
#run_and_animate(params)

## FD
m = central_fdm(2, 1);
ad = AutoFiniteDifferences(m)
function manual_fd_jacobian(params, h)
    values = []
    jacobians = []
    for i in 1:length(params)
        params_plus = copy(params)
        params_minus = copy(params)
        params_plus[i] += h
        params_minus[i] -= h
        value_plus = run_model(params_plus, gradient_horizon = 10000)
        value_minus = run_model(params_minus, gradient_horizon = 10000)
        jacobian = (value_plus - value_minus) / (2 * h)
        push!(values, value_plus)
        push!(jacobians, jacobian)
    end
    return hcat(values...), hcat(jacobians...)
end
#value_fd, jacobian_fd = get_value_and_jacobian_of_average(run_model, params, ad, 1);
value_fd, jacobian_fd = manual_fd_jacobian(params, 1.0)

function get_ad_jacobian(params; n_samples = 100, gradient_horizon)
    values = []
    jacobians = []
    for i in 1:n_samples
        value, jacobian = DifferentiationInterface.value_and_jacobian(
            x -> run_model(x, gradient_horizon = gradient_horizon), AutoForwardDiff(), params)
        push!(values, value)
        push!(jacobians, jacobian)
    end
    return hcat(values...), hcat(jacobians...)
end
values_ad_0, jacobians_ad_0 = get_ad_jacobian(params, n_samples = 1, gradient_horizon = 1);
values_ad_1, jacobians_ad_1 = get_ad_jacobian(params, n_samples = 1, gradient_horizon = 101);
#jacobians_st = hcat([derivative_estimate(run_model, params)[1] for _ in 1:1000]...)

## plot the jacobian
n_plots = 1#size(jacobian, 2)
fig, ax = plt.subplots(1, n_plots, figsize = (6, 3))
titles = ["sugar_regeneration_rate"]
for i in 1:n_plots
    if i == 1
        axis = ax
    else
        axis = ax[i]
    end
    #axis.plot(jacobians_ad[:, i], label = "AD")
    #axis.boxplot(jacobians_ad, showfliers = false)
    #axis.plot(mean(jacobians_ad, dims = 2), label = "AD")
    #axis.plot(mean(jacobians_st, dims = 2), label = "Striple")
    ax.plot(mean(jacobians_ad_0, dims = 2), label = "AD 1")
    #ax.plot(mean(jacobians_ad_1, dims = 2), label = "AD 101")
    #axis.plot(jacobian_fd[:, i], color = "black", label = "FD", linestyle = "--")
    axis.set_title(titles[i])
    #axis.set_yscale("log")
    axis.legend()
end
fig


##
jacobians = jacobians_ad
partials = [jacobians[i+1] ./ jacobians[i] for i in 1:size(jacobians, 1)-1][3:end]
partials_weighted = []
gamma = 0.9
for i in 1:size(partials, 1)
    val = 1.0
    for j in 1:i-1
        val *= partials[j] * gamma^(i-j)
    end
    val += partials[i]
    push!(partials_weighted, val)
end
fig, ax = plt.subplots()
#ax.plot(partials)
#ax.plot(mean(jacobians_ad, dims = 2))
ax.plot(partials, label = "partials")

ax.plot(partials_weighted, "o-", markersize = 2, label = "weighted partials")
ax2 = ax.twinx()
ax2.plot(values_ad, label = "values", color = "red")
ax.plot(jacobian_fd[:, 1], color = "black", linestyle = "--", label = "FD")
ax.plot(mean(jacobians_ad, dims = 2), label = "AD")
ax.set_xlabel("Timestep")
ax.set_ylabel("Partial derivative")
ax.set_yscale("log")
ax.set_ylim(1e-3, 1e10)
ax.legend()
fig

##
partials_xx = [1.0]
j0s = jacobians_ad_0[:][3:end]
j1s = jacobians_ad_1[:][3:end]
for i = 2:length(j0s)
    pxx = (j1s[i] - j0s[i]) / (partials_xx[i-1])
    push!(partials_xx, pxx)
end
##
partials_weighted = [j0s[1]]
gamma = 0.2
for i in 2:length(j0s)
    pweighted = j0s[i] + partials_xx[i] * partials_weighted[i-1] ^ gamma
    push!(partials_weighted, pweighted)
end

fig, ax = plt.subplots()
#ax.plot(partials_xx)
ax.plot(j0s)
ax.plot(partials_weighted)
ax.plot(j1s)
ax.set_yscale("log")
fig


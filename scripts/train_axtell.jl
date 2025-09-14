# Example script where we train the Axtell model using a simple gradient descent
using DiffABM
using StatsBase
using PyPlot
using Flux
using DifferentiationInterface
using Optimisers
using Random
using Distributions
using Random
Random.seed!(42)

const DI = DifferentiationInterface

# Set up the Axtell model parameters
"""
Here bounds represents the parameters for a beta distribution.
Creates parameters for the Axtell firms ABM with specified bounds for agent initialization.
"""
function make_axtell_params(theta_bounds, initial_effort_bounds, a_bounds, b_bounds)
    n_agents = 500
    n_timesteps = 30
    update_rate = 0.25  # Rate at which agents update their strategies
    delta_t = 1.0
    gradient_horizon = 10000  # Time horizon for gradient computation
    n_neighbors = 4  # Number of neighbors each agent observes

    # Initialize agents with random parameters drawn from specified bounds
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
# Define parameter bounds for agent initialization (beta distribution parameters)
theta_bounds = [1.0, 2.0]
initial_effort_bounds = [3.0, 4.0]
a_bounds = [5.0, 6.0]
b_bounds = [7.0, 8.0]

# Create ABM parameters and run the model to generate "true" data
abm_params = make_axtell_params(theta_bounds, initial_effort_bounds, a_bounds, b_bounds)
x = abm_run(abm_params)  # x contains [mean_effort, mean_firm_size, mean_firm_output] over time

# Plot the "true" data from the ABM simulation
fig, axs = plt.subplots(1, 3, figsize = (10, 3))
axs[1].plot(x[1, :])
axs[1].set_title("Mean effort per time-step")
axs[2].plot(x[2, :])
axs[2].set_title("Mean firm size per time-step")
axs[3].plot(x[3, :])
axs[3].set_title("Mean firm output per time-step")
for ax in axs
    ax.set_xlabel("timestep")
end
fig

# Training the model using gradient descent

function compute_loss(log_params_flat, y, restruct_f)
    # Loss function: minimize MSE between predicted and true mean firm output
    params_flat = exp.(log_params_flat)  # Transform from log space to ensure positive parameters
    x = abm_run(restruct_f(params_flat))  # Run ABM with current parameters
    return mean((x[3, :] .- y[3, :]) .^ 2)  # MSE on mean firm output (3rd row)
end

function train_model(y, n_epochs)
    # Extract parameter structure from the original ABM parameters
    _, restruct_f = Flux.destructure(abm_params)

    # Initialize with random parameters in log space (8 parameters total)
    log_params_flat = 5 .* rand(8)

    # Set up Adam optimizer with learning rate 0.01
    rule = Optimisers.Adam(0.01)
    state = Optimisers.setup(rule, log_params_flat)

    # Track loss and parameter evolution during training
    loss_history = []
    param_history = []

    # Main training loop
    for i in 1:n_epochs
        # Compute loss and gradients using automatic differentiation
        loss, grad = DifferentiationInterface.value_and_gradient(
            compute_loss, AutoForwardDiff(), log_params_flat, DI.Constant(y), DI.Constant(restruct_f))

        # Store current loss and parameters
        push!(loss_history, loss)
        push!(param_history, copy(log_params_flat))

        # Update parameters using optimizer
        state, log_params_flat = Optimisers.update!(state, log_params_flat, grad)
    end
    return loss_history, param_history
end

# Run the training for 2000 epochs
loss_history, param_history = train_model(x, 5000);
loss_history = vcat(loss_history...)  # Convert to flat array
param_history = hcat(param_history...)  # Convert to matrix (params × epochs)

# Visualize training results
fig, ax = plt.subplots(1, 2, figsize = (10, 5));

# Plot loss convergence
ax[1].plot(loss_history)
ax[1].set_yscale("log")
ax[1].set_title("Loss history")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")

# Compare initial vs final parameter performance against true data
_, restruct_f = Flux.destructure(abm_params)

# Run ABM with initial random parameters
params_initial = exp.(param_history[:, 1])
x_initial = abm_run(restruct_f(params_initial))

# Run ABM with final trained parameters
params_final = exp.(param_history[:, end])
x_final = abm_run(restruct_f(params_final))

# Plot comparison of trajectories
ax[2].plot(x_initial[3, :], color = "C0", label = "Initial")
ax[2].plot(x_final[3, :], color = "C1", label = "Final")
ax[2].plot(x[3, :], color = "black", label = "True")
ax[2].legend()
ax[2].set_yscale("log")
ax[2].set_title("Mean firm output per time-step")
ax[2].set_xlabel("Timestep")
ax[2].set_ylabel("Mean firm output")
fig.savefig("axtell_training.png")
fig
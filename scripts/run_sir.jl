using DiffABM
using ForwardDiff
using Flux
using PyPlot

##

function get_params()
    n_agents = 10000;
    delta_t = 1.0;
    n_timesteps = 60;
    model_params = (
        initial_infected = [0.005],
        beta = [0.4],
        gamma = [0.05],
        social_distancing_start_time = [10.0],
        social_distancing_end_time = [35.0],
        alpha = [0.4],
        quarantine_start_time = [15.0],
        quarantine_end_time = [25.0],
        p_quarantine = [0.7],
    );
    neighbours = [setdiff(rand(1:n_agents, 10), [i]) for i in 1:n_agents];

    params = SIRNonVecParams(
        neighbours = neighbours,
        model_params = model_params,
        n_agents = n_agents,
        delta_t = delta_t,
        n_timesteps = n_timesteps,
        discrete_sampler = ST(),
        use_policies = true
    )
    return params
end

params = get_params()
x = abm_run(params);

fig, ax = plt.subplots(1, 2, figsize=(10, 5));
ax[1].plot(x[1, :]);
ax[2].plot(x[2, :]);
fig

## Jacobian with respect to parameters

params_flat, restruct_f = Flux.destructure(params)
function run_for_p(p)
    return abm_run(restruct_f(p))
end
jacobian = ForwardDiff.jacobian(run_for_p, params_flat)

function plot_jacobian(jacobian)
    fig, ax = plt.subplots(3, 3, figsize=(10, 5));
    jacobian_idx = 1
    for i in 1:3
        for j in 1:3
            ax[i, j].plot(jacobian[:, jacobian_idx])
            jacobian_idx += 1
        end
    end
    fig
end

plot_jacobian(jacobian)
using DiffABM
using ForwardDiff
using Flux
using PyPlot
using BenchmarkTools
using Profile, PProf

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
    #neighbours = [setdiff(1:n_agents, [i]) for i in 1:n_agents];
    neighbours = [setdiff(rand(1:n_agents, 10), [i]) for i in 1:n_agents];

    params = SIRNonVecParams(
        neighbours = neighbours,
        model_params = model_params,
        n_agents = n_agents,
        delta_t = delta_t,
        n_timesteps = n_timesteps,
        discrete_sampler = ST()
    )
    return params
end

params = get_params()
x = abm_run(params);

fig, ax = plt.subplots(1, 2, figsize=(10, 5));
ax[1].plot(x[1, :]);
ax[2].plot(x[2, :]);
fig

##
function run_for_p(p)
    _, f = Flux.destructure(params)
    return abm_run(f(p))
end
p_flat = vcat(Flux.params(model_params)...)

##
@time run_for_p(p_flat)
##
Profile.clear()
@profile run_for_p(p_flat)
pprof()

##
@time ForwardDiff.jacobian(run_for_p, p_flat)



##
using JLD2
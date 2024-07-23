export BrockHommesParams

struct BrockHommesParams{T}
    n_timesteps::Int64
    params::Vector{T}
end

function compute_u(x, g, b, R)
    t1 = x[end] - R * x[end - 1]
    t21 = g .* x[end - 2] .+ b
    t22 = R * x[end - 1]
    t2 = t21 .- t22
    return t1 .* t2
end

function abm_step(params::BrockHommesParams, x, t)
    p = params.params
    beta = p[1]
    g = p[2:5]
    b = p[6:9]
    sigma = p[10]
    r = p[11]
    R = 1.0 + r

    epsilon = rand(Normal(0.0, 1.0))
    u_h = compute_u(x, g, b, R)
    strategy = Flux.softmax(beta * u_h)
    mean = sum(strategy .* (g .* x[end] + b))
    return (mean + epsilon * sigma) / R
end

function abm_run(params::BrockHommesParams)
    x = zeros(3)
    for t in 4:(params.n_timesteps)
        x_t = abm_step(params, x, t)
        x = vcat(x, x_t)
    end
    return x
end

function abm_logpdf(params::BrockHommesParams, y)
    beta = params[1]
    g = params[2:5]
    b = params[6:9]
    sigma = params[10]
    r = params[11]
    R = 1.0 + r
    scale = sigma / R
    lp = 0.0
    n_timesteps = bh.n
    for t in 4:n_timesteps
        u_h = compute_u(y[1:t], g, b, R)
        strategy = Flux.softmax(beta * u_h)
        mean = sum(strategy .* (g .* y[t - 1] + b)) / R
        lp += logpdf(Normal(mean, scale), y[t])
    end
    return lp
end
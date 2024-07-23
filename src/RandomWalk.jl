export RandomWalkParams

struct RandomWalkParams{T, S}
    n_timesteps::Int64
    discrete_sampler::S
    p::T
end

function step(params::RandomWalkParams{T}, x, t) where {T}
    delta_x = 2 * sample_bernoulli(params.discrete_sampler, [params.p]) - 1
    x_t = x[end] + delta_x
    return x_t
end

function run(params::RandomWalkParams{T}) where {T}
    x = [zero(T)]
    for t in 2:params.n_timesteps
        x_t = step(params, x, t)
        x = vcat(x, x_t)
    end
    return x
end

logpdf(::RandomWalkParams{T}, y) = throw("Intractable")
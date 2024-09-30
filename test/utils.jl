function get_function_average(f, x, n_samples, state)
    copy!(Random.default_rng(), state)
    return sum(fetch.([Threads.@spawn f(x) for _ in 1:n_samples])) / n_samples
end

function get_value_and_jacobian_of_average(f, x, ad, n_samples)
    state = copy(Random.default_rng())
    return DifferentiationInterface.value_and_jacobian(
        x -> get_function_average(f, x, n_samples, state), ad, x)
end
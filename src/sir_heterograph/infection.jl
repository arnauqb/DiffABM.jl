export GammaInfection, ConstantInfection

abstract type Infection end
struct GammaInfection <: Infection
    μ::Float64
    σ::Float64
end
GammaInfection(config) = GammaInfection(config[:μ], config[:σ])
struct ConstantInfection <: Infection end
ConstantInfection(config) = ConstantInfection()

function (infection::GammaInfection)(time_since_infection)
    ret = gamma_pdf(infection.μ, infection.σ, time_since_infection + 1e-10)
    # clamp to avoid numerical issues
    ret = clamp(ret, 1e-8, 1.0)
    return ret
end

function (infection::ConstantInfection)(time_since_infection)
    return 1.0
end

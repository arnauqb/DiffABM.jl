export SocialDistancing, SocialDistancingPolicies, Quarantine, QuarantinePolicies, Policies

abstract type AbstractPolicy end
abstract type AbstractPolicies end
Base.getindex(ps::AbstractPolicies, i) = ps.policies[i]

struct SocialDistancing{T,U,V} <: AbstractPolicy
    start_time::Vector{V}
    end_time::Vector{V}
    alphas_by_venue::Dict{U,Vector{T}}
end
@functor SocialDistancing (start_time, end_time, alphas_by_venue)
SocialDistancing(config::Dict) = SocialDistancing(
    [config[:start_time]], [config[:end_time]], Dict(config[:alphas_by_venue]))

struct SocialDistancingPolicies <: AbstractPolicies
    policies::Vector{SocialDistancing}
end
@functor SocialDistancingPolicies (policies, )
SocialDistancingPolicies() = SocialDistancingPolicies(SocialDistancing[])
function SocialDistancingPolicies(config::Dict)
    policies = SocialDistancing[]
    for (_, value) in config
        push!(policies, SocialDistancing(value))
    end
    return SocialDistancingPolicies(policies)
end


function (p::SocialDistancing)(x, time, venue)
    mask = differentiable_step(p.start_time[1], p.end_time[1], time)
    return @. x * (mask * p.alphas_by_venue[venue] + (1.0 - mask))
end

function (p::SocialDistancingPolicies)(x, time, venue)
    for policy in p.policies
        x = policy(x, time, venue)
    end
    return x
end

struct Quarantine{T,V} <: AbstractPolicy
    start_time::Vector{V}
    end_time::Vector{V}
    p::Vector{T}
end
@functor Quarantine (start_time, end_time, p)
Quarantine(config) = Quarantine([config[:start_time]], [config[:end_time]], [config[:p]])

struct QuarantinePolicies <: AbstractPolicies
    policies::Vector{Quarantine}
end
@functor QuarantinePolicies (policies, )
QuarantinePolicies() = QuarantinePolicies(Quarantine[])
function QuarantinePolicies(config::Dict)
    policies = Quarantine[]
    for (_, value) in config
        push!(policies, Quarantine(value))
    end
    return QuarantinePolicies(policies)
end

function (p::Quarantine)(sampler, transmission, time)
    mask = differentiable_step(p.start_time[1], p.end_time[1], time)
    quarantine_probs = ones(length(transmission)) .* p.p[1]
    does_quarantine = sample_bernoulli(sampler, quarantine_probs)
    return @. transmission * (mask * does_quarantine + (1.0 - mask))
end

function (p::QuarantinePolicies)(sampler, transmission, time)
    for policy in p.policies
        transmission = policy(sampler, transmission, time)
    end
    return transmission
end

struct Policies
    social_distancing::SocialDistancingPolicies
    quarantine::QuarantinePolicies
end
@functor Policies (social_distancing, quarantine)
Policies() = Policies(SocialDistancingPolicies(), QuarantinePolicies())
function Policies(config::Dict)
    sd = SocialDistancingPolicies(config[:social_distancing])
    qs = QuarantinePolicies(config[:quarantine])
    return Policies(sd, qs)
end
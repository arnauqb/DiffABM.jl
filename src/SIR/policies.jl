export SocialDistancing, SocialDistancingPolicies, Quarantine, QuarantinePolicies, Policies

abstract type AbstractPolicy end
abstract type AbstractPolicies end
Base.getindex(ps::AbstractPolicies, i) = ps.policies[i]

struct SocialDistancing{T,U,V} <: AbstractPolicy
    start_time::V
    end_time::V
    alphas_by_venue::Dict{U,T}
end
SocialDistancing(config::Dict) = SocialDistancing(config[:start_time], config[:end_time], config[:alphas_by_venue])

struct SocialDistancingPolicies <: AbstractPolicies
    policies::Vector{SocialDistancing}
end
SocialDistancingPolicies() = SocialDistancingPolicies(SocialDistancing[])
function SocialDistancingPolicies(config::Dict)
    policies = SocialDistancing[]
    for (_, value) in config
        push!(policies, SocialDistancing(value))
    end
    return SocialDistancingPolicies(policies)
end


function (p::SocialDistancing)(x, time, venue)
    if p.start_time <= time < p.end_time && haskey(p.alphas_by_venue, venue)
        return x .* p.alphas_by_venue[venue]
    else
        return x
    end
end

function (p::SocialDistancingPolicies)(x, time, venue)
    for policy in p.policies
        x = policy(x, time, venue)
    end
    return x
end

struct Quarantine{T,V} <: AbstractPolicy
    start_time::V
    end_time::V
    p::T
end
Quarantine(config) = Quarantine(config[:start_time], config[:end_time], config[:p])

struct QuarantinePolicies <: AbstractPolicies
    policies::Vector{Quarantine}
end
QuarantinePolicies() = QuarantinePolicies(Quarantine[])
function QuarantinePolicies(config::Dict)
    policies = Quarantine[]
    for (_, value) in config
        push!(policies, Quarantine(value))
    end
    return QuarantinePolicies(policies)
end

function (p::Quarantine)(sampler, transmission, time)
    if p.start_time <= time < p.end_time
        quarantine_probs = ones(length(transmission)) .* p.p
        return sampler(quarantine_probs) .* transmission
    else
        return transmission
    end
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
Policies() = Policies(SocialDistancingPolicies(), QuarantinePolicies())
function Policies(config::Dict)
    sd = SocialDistancingPolicies(config[:social_distancing])
    qs = QuarantinePolicies(config[:quarantine])
    return Policies(sd, qs)
end
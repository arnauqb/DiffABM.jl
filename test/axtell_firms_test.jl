using DiffABM
using Test

@testset "AxtellFirms" begin
    @testset "utility and effort functions" begin
        a = 1.0
        b = 1.0
        theta = 0.5
        agent_effort = 0.5
        firm_output = 2.0
        size = 2.0
        @testset "CobbDouglasUtility" begin
            utility_function = DiffABM.CobbDouglasUtility()
            @test utility_function(firm_output, size, theta, agent_effort) ==
                  (2.0 / 2.0)^0.5 * (1.0 - 0.5)^(1.0 - 0.5)
        end
        @testset "compute_group_effort_from_output" begin
            # this solves the equation O = a(E + e) + b(E + e)^2
            # so we can check if the result is correct
            E = DiffABM.compute_group_effort_from_output(firm_output, agent_effort, a, b)
            @test firm_output ≈ a * (E + agent_effort) + b * (E + agent_effort)^2
        end
        @testset "compute_optimal_effort" begin
            # this effort maximies the ulitiy function
            utility_function = DiffABM.CobbDouglasUtility()
            E = DiffABM.compute_group_effort_from_output(firm_output, agent_effort, a, b)
            utility_values = []
            for agent_e in 0:0.01:1.0
                total_effort = E + agent_e
                output = DiffABM.compute_firms_output(total_effort, a, b)
                push!(utility_values, utility_function(output, size, theta, agent_e))
            end
            best_e = DiffABM.compute_optimal_effort(theta, E, a, b)
            optimal_utility = utility_function(firm_output, size, theta, best_e)
            @test all(utility_values .<= optimal_utility)
        end
        @testset "compute_firms_output" begin
            @test DiffABM.compute_firms_output(10, a, b) == a * 10 + b * 10^2
        end
    end
end
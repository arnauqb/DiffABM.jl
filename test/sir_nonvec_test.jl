using DiffABM
using Test

@testset "SIR NonVec" begin
    @testset "initialize" begin
        S, I, R, delta_I, delta_R = DiffABM.initialize(ST(), 100, 0.1)
        @test sum(S) + sum(I) == 100
        @test sum(I) > 0 && sum(I) < 100
        @test all(R .== 0)
    end

    @testset "is_active" begin
        @test DiffABM.is_active(5.0, 3.0, 8.0) > 0.5
        @test DiffABM.is_active(2.0, 3.0, 8.0) < 0.5
        @test DiffABM.is_active(9.0, 3.0, 8.0) < 0.5
    end

    @testset "apply_social_distancing" begin
        beta = 0.5
        alpha = 0.3
        # Inside period
        result = DiffABM.apply_social_distancing(beta, 5.0, 3.0, 8.0, alpha)
        @test result < beta
        # Outside period
        result = DiffABM.apply_social_distancing(beta, 10.0, 3.0, 8.0, alpha)
        @test result ≈ beta
    end

    @testset "abm_run with different samplers" begin
        neighbours = [[1, 2], [1, 2]]
        model_params = (beta=[0.3], gamma=[0.1], initial_infected=[0.5],
                       quarantine_start_time=[10.0], quarantine_end_time=[15.0], 
                       p_quarantine=[0.0], social_distancing_start_time=[10.0],
                       social_distancing_end_time=[15.0], alpha=[1.0])
        
        for sampler in [GS(0.1), SAD(), ST()]
            params = SIRNonVecParams(neighbours=neighbours, use_policies=false,
                                   model_params=model_params, n_agents=2,
                                   delta_t=1.0, n_timesteps=5, 
                                   discrete_sampler=sampler)
            result = abm_run(params)
            @test size(result) == (2, 5)
            @test all(result .>= 0)
        end
    end

    @testset "policies effect" begin
        neighbours = [collect(1:4) for _ in 1:4]
        model_params = (beta=[0.8], gamma=[0.2], initial_infected=[0.5],
                       quarantine_start_time=[2.0], quarantine_end_time=[4.0],
                       p_quarantine=[1.0], social_distancing_start_time=[2.0],
                       social_distancing_end_time=[4.0], alpha=[0.1])
        
        params_no_pol = SIRNonVecParams(neighbours=neighbours, use_policies=false,
                                       model_params=model_params, n_agents=4,
                                       delta_t=1.0, n_timesteps=6,
                                       discrete_sampler=ST())
        params_with_pol = SIRNonVecParams(neighbours=neighbours, use_policies=true,
                                         model_params=model_params, n_agents=4,
                                         delta_t=1.0, n_timesteps=6,
                                         discrete_sampler=ST())
        
        result_no_pol = abm_run(params_no_pol)
        result_with_pol = abm_run(params_with_pol)
        
        @test size(result_no_pol) == size(result_with_pol) == (2, 6)
        @test sum(result_with_pol[1, 3:4]) <= sum(result_no_pol[1, 3:4])  # less infection during policy
    end
end
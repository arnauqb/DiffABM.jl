using Test
using DelimitedFiles
using DiffABM

function make_sugarscape_params(vision_probs, metabolic_rate_bounds, wealth_bounds)
    board_length = 50
    n_agents = 100
    n_timesteps = 100
    board = readdlm("scripts/sugar-map.txt")
    board_initializer = GeneratedBoard(board_length, board[:])
    discrete_sampler = ST()
    neighborhood = VonNeumannNeighborhood()
    sugar_regeneration_rate = 1.0
    gradient_horizon = 10000
    agent_initializer = RandomAgentInitializer(
        board_length,
        vision_distribution_probs = vision_probs,
        metabolic_rate_bounds = metabolic_rate_bounds,
        wealth_bounds = wealth_bounds,
        neighborhood = neighborhood,
        discrete_sampler = discrete_sampler,
        vision_spacing = 1)
    smoothing = GaussianSmoothing(1.0)
    sugarscape = SugarScapeParams(
        board_initializer, agent_initializer, board_length,
        n_agents, n_timesteps, [sugar_regeneration_rate], gradient_horizon, smoothing)
    return sugarscape
end

@testset "SugarScape" begin
    @testset "generate_vision_matrices" begin
        # Test VonNeumann neighborhood
        max_vision = 2
        von_neumann_matrices = DiffABM.generate_vision_matrices(
            VonNeumannNeighborhood(), max_vision)

        # Should return the correct number of matrices (one per vision radius)
        @test length(von_neumann_matrices) == max_vision

        # Check dimensions of matrices
        @test size(von_neumann_matrices[1]) == (2 * max_vision + 1, 2 * max_vision + 1)

        # For vision radius 1 in VonNeumann, should have 5 ones (center + 4 adjacent)
        expected_vision1 = zeros(5, 5)
        expected_vision1[3, 3] = 1  # Center
        expected_vision1[2, 3] = 1  # North
        expected_vision1[3, 2] = 1  # West
        expected_vision1[4, 3] = 1  # South
        expected_vision1[3, 4] = 1  # East
        @test von_neumann_matrices[1] == expected_vision1

        # For vision radius 2 in VonNeumann, should include all cells with Manhattan distance <= 2
        expected_vision2 = zeros(5, 5)
        expected_vision2[3, 3] = 1  # Center
        expected_vision2[2, 3] = 1  # North
        expected_vision2[3, 2] = 1  # West
        expected_vision2[4, 3] = 1  # South
        expected_vision2[3, 4] = 1  # East
        expected_vision2[1, 3] = 1  # North-North
        expected_vision2[3, 1] = 1  # West-West
        expected_vision2[5, 3] = 1  # South-South
        expected_vision2[3, 5] = 1  # East-East
        expected_vision2[2, 2] = 1  # North-West
        expected_vision2[2, 4] = 1  # North-East
        expected_vision2[4, 2] = 1  # South-West
        expected_vision2[4, 4] = 1  # South-East
        @test von_neumann_matrices[2] == expected_vision2

        # Test Moore neighborhood
        moore_matrices = DiffABM.generate_vision_matrices(MooreNeighborhood(), max_vision)

        # Should return the correct number of matrices
        @test length(moore_matrices) == max_vision

        # For vision radius 1 in Moore, should have 9 ones (center + 8 surrounding)
        expected_moore1 = zeros(5, 5)
        for i in 2:4
            for j in 2:4
                expected_moore1[i, j] = 1
            end
        end
        @test moore_matrices[1] == expected_moore1

        # For vision radius 2 in Moore, should include all cells with Chebyshev distance <= 2
        expected_moore2 = zeros(5, 5)
        for i in 1:5
            for j in 1:5
                expected_moore2[i, j] = 1
            end
        end
        @test moore_matrices[2] == expected_moore2
    end

    @testset "compute_move" begin
        # Create a simple board with sugar values
        board_size = 5
        board = zeros(Float64, board_size, board_size)

        # Place sugar at specific locations with a clear winner (North)
        board[2, 3] = 10.0  # North of center (highest value)
        board[3, 4] = 5.0   # East of center
        board[4, 3] = 1.0   # South of center
        board[3, 2] = 2.0   # West of center

        # Create an occupied matrix (1.0 means occupied, 0.0 means free)
        occupied = zeros(Float64, board_size, board_size)

        # Create a vision matrix for vision radius 1 (VonNeumann)
        vision_matrix = DiffABM.generate_vision_matrices(VonNeumannNeighborhood(), 1)[1]

        # Create an agent at the center of the board
        agent = DiffABM.SugarSeeker(
            id = 1,
            vision_matrix = vision_matrix,
            metabolic_rate = 1.0,
            x = 3.0,
            y = 3.0,
            alive = 1.0,
            wealth = 10.0
        )
        occupied[3, 3] = 1.0

        # Compute the move
        move_onehot = DiffABM.compute_move(board, agent, occupied)

        # The agent should move to the position with the highest sugar
        # In this case, it's the North position with 10.0 sugar
        # Since North is at index 2 in the flattened 3x3 grid, we expect a 1.0 at that position
        expected_move = zeros(Float64, 9)
        expected_move[2] = 1.0  # North position (index 2 in the flattened 3x3 grid)

        # Check that the move_onehot has a 1.0 at the North position
        @test findmax(move_onehot)[2] == 2

        # Test with occupied positions
        # Now make the highest sugar position (North) occupied
        occupied[2, 3] = 1.0  # North is occupied

        move_onehot2 = DiffABM.compute_move(board, agent, occupied)

        # The agent should move to the position with the highest sugar that is not occupied
        # In this case, it's the East position with 5.0 sugar
        # Since East is at index 6 in the flattened 3x3 grid, we expect a 1.0 at that position
        expected_move2 = zeros(Float64, 9)
        expected_move2[6] = 1.0  # East position (index 6 in the flattened 3x3 grid)

        # Check that the move_onehot has a 1.0 at the East position
        @test findmax(move_onehot2)[2] == 6

        # Test with only one free position
        occupied3 = ones(Float64, board_size, board_size)
        occupied3[4, 3] = 0.0  # Only South is free

        move_onehot3 = DiffABM.compute_move(board, agent, occupied3)

        # The agent should move to the only free position
        # In this case, it's the South position
        # Since South is at index 8 in the flattened 3x3 grid, we expect a 1.0 at that position
        expected_move3 = zeros(Float64, 9)
        expected_move3[8] = 1.0  # South position (index 8 in the flattened 3x3 grid)

        # Check that the move_onehot has a 1.0 at the South position
        @test findmax(move_onehot3)[2] == 8

        # Test with all positions occupied
        occupied4 = ones(Float64, board_size, board_size)

        move_onehot4 = DiffABM.compute_move(board, agent, occupied4)

        # The agent should stay in place (center) as all positions are occupied
        # All scores should be 0, so the agent stays at its current position
        @test sum(move_onehot4) ≈ 1.0  # The sum should be 1.0 (one-hot encoding)

        # Test with a tie (should be handled by random_argmax)
        board_tie = zeros(Float64, board_size, board_size)
        board_tie[2, 3] = 5.0  # North
        board_tie[3, 4] = 5.0  # East

        occupied_tie = zeros(Float64, board_size, board_size)

        move_onehot_tie = DiffABM.compute_move(board_tie, agent, occupied_tie)

        # The agent should move to either North or East (both have 5.0 sugar)
        # We can't predict which one due to the random nature of differentiable_argmax in case of ties
        # So we check that one of them has the highest value
        max_idx_tie = findmax(move_onehot_tie)[2]
        @test max_idx_tie == 2 || max_idx_tie == 6  # North (index 2) or East (index 6)
    end
    @testset "model jacobian with ForwardDiff" begin
        n_timesteps = 100
        params = make_sugarscape_params([0.5, 0.5], [1.0, 2.0], [1.0, 2.0])
        params_flat, restruct_f = Flux.destructure(params)
        function run_for_p(p)
            x = abm_run(restruct_f(p))
            wealth_per_timestep = [sum(x[4][t]) for t in 1:n_timesteps]
            return wealth_per_timestep
        end
        jacobian = ForwardDiff.jacobian(run_for_p, params_flat)
        @test size(jacobian) == (100, 6)
        @test all(isfinite.(jacobian))
    end
end

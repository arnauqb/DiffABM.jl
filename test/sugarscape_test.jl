using Test
using DiffABM

@testset "Von Neumann Neighborhood" begin
    board_length = 5
    i = 3
    j = 3
    vision = 2
    vn = VonNeumannNeighborhood(board_length, i, j, vision)
    correct_sequence = [(3,4), (3,5), (2,3), (1,3), (4,3), (5,3), (3,2), (3,1)]
    for (pos, true_pos) in zip(vn, correct_sequence)
        @test pos == true_pos
    end
end
using Test
using DiffABM

@testset "Von Neumann Neighborhood" begin
	board_length = 5
	i = 3
	j = 3
	vision = 2
	vn = VonNeumannNeighborhood(board_length, vision)
	correct_sequence = [(3, 4), (3, 5), (3, 2), (3, 1), (4, 3), (5, 3), (2, 3), (1, 3)]
	for (pos, true_pos) in zip(DiffABM.iterate(vn, i, j, vision), correct_sequence)
		@test pos == true_pos
	end
end

@testset "Moore Neighborhood" begin
	board_length = 5
	i = 3
	j = 3
	vision = 2
	moore = MooreNeighborhood(board_length, vision)
	correct_sequence = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 1), (3, 2), (3, 4), (3, 5),
        (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)
	]
	for (pos, true_pos) in zip(DiffABM.iterate(moore, i, j, vision), correct_sequence)
		@test pos == true_pos
	end
end

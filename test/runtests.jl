using DiffABM
using Test

@testset "DiffABM.jl" begin
    # Write your tests here.
    include("game_of_life_test.jl")
    include("brock_hommes_test.jl")
    include("sir_test.jl")
    include("random_walk_test.jl")
end

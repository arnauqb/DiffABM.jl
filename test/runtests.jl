using DiffABM
using Test

@testset "DiffABM.jl" begin
    # Write your tests here.
    include("diff_utils_test.jl")
    include("sir_test.jl")
    include("random_walk_test.jl")
    include("axtell_firms_test.jl")
    include("sugar_scape_test.jl")
end

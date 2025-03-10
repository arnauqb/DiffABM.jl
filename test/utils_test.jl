using Test
using DiffABM: wrap_index

@testset "wrap_index" begin
    # Test normal case (within bounds)
    @test wrap_index(5, 3, 4) == (3, 4)
    
    # Test wrapping around horizontally
    @test wrap_index(5, 6, 3) == (1, 3)
    @test wrap_index(5, 0, 3) == (5, 3)
    
    # Test wrapping around vertically
    @test wrap_index(5, 3, 6) == (3, 1)
    @test wrap_index(5, 3, 0) == (3, 5)
    
    # Test wrapping around both dimensions
    @test wrap_index(5, 6, 6) == (1, 1)
    @test wrap_index(5, 0, 0) == (5, 5)
    
    # Test with larger wrapping
    @test wrap_index(5, 8, 7) == (3, 2)
    @test wrap_index(5, -2, -3) == (3, 2)
    
    # Test with different board sizes
    @test wrap_index(3, 4, 5) == (1, 2)
    @test wrap_index(10, 12, 15) == (2, 5)
end

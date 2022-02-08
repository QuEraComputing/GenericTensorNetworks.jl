using GraphTensorNetworks, Test, Graphs

@testset "mis compactify" begin
    g = SimpleGraph(6)
    for (i,j) in [(1,2), (2,3), (4,5), (5,6), (1,6)]
        add_edge!(g, i, j)
    end
    g = IndependentSet(g, openvertices=[1,4,6,3])
    m = solve(g, SizeMax())
    @test m isa Array{Tropical{Float64}, 4}
    @test count(!iszero, m) == 12
    mis_compactify!(m)
    @test count(!iszero, m) == 3
end
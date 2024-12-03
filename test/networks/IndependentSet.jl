using GenericTensorNetworks, Test, Graphs

@testset "mis compactify" begin
    g = SimpleGraph(6)
    for (i,j) in [(1,2), (2,3), (4,5), (5,6), (1,6)]
        add_edge!(g, i, j)
    end
    g = GenericTensorNetwork(IndependentSet(g); openvertices=[1,4,6,3])
    m = solve(g, SizeMax())
    @test m isa Array{Tropical{Float64}, 4}
    @test count(!iszero, m) == 12
    m1 = mis_compactify!(copy(m))
    @test count(!iszero, m1) == 3
    potential = zeros(Float64, 4)
    m2 = mis_compactify!(copy(m); potential)
    @test count(!iszero, m2) == 1
    @test get_weights(g) == UnitWeight(nv(g.problem.graph))
    @test get_weights(set_weights(g, fill(3, 6))) == fill(3, 6)
end

@testset "empty graph" begin
    g = SimpleGraph(4)
    pb = GenericTensorNetwork(IndependentSet(g))
    @test solve(pb, SizeMax()) !== 4
end
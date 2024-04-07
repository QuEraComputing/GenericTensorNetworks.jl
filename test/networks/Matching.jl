using Test, GenericTensorNetworks, Graphs
using GenericTensorNetworks: solutions

@testset "enumerating - matching" begin
    g = smallgraph(:petersen)
    code = GenericTensorNetwork(Matching(g); optimizer=GreedyMethod(), fixedvertices=Dict())
    res = solutions(code, CountingTropicalF64; all=true)[]
    @test res.n == 5
    @test length(res.c.data) == 6
    code = GenericTensorNetwork(Matching(g); optimizer=GreedyMethod(), fixedvertices=Dict((1,2)=>1))
    @test get_weights(code) == UnitWeight()
    @test get_weights(chweights(code, fill(3, 15))) == fill(3, 15)
    res = solutions(code, CountingTropicalF64; all=true)[]
    @test res.n == 5
    k = findfirst(x->x==(1,2), labels(code))
    @test length(res.c.data) == 2 && res.c.data[1][k] == 1 && res.c.data[2][k] == 1
end

@testset "match polynomial" begin
    g = SimpleGraph(7)
    for (i,j) in [(1,2),(2,3),(3,4),(4,5),(5,6),(6,1),(1,7)]
        add_edge!(g, i, j)
    end
    @test graph_polynomial(GenericTensorNetwork(Matching(g)), Val(:polynomial))[] == Polynomial([1,7,13,5])
    g = smallgraph(:petersen)
    @test graph_polynomial(GenericTensorNetwork(Matching(g)), Val(:polynomial))[].coeffs == [6, 90, 145, 75, 15, 1][end:-1:1]
end

@testset "weighted matching" begin
    g = smallgraph(:petersen)
    problem = GenericTensorNetwork(Matching(g, fill(2, 15)))
    @test solve(problem, SizeMax())[].n == 10
    res = solve(problem, SingleConfigMax())[].c.data
    @test is_matching(g, res)
    @test count_ones(res) * 2 == 10
end
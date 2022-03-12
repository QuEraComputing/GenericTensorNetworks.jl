using Test, GraphTensorNetworks, Graphs

@testset "enumerating - matching" begin
    g = smallgraph(:petersen)
    code = Matching(g; optimizer=GreedyMethod())
    res = solutions(code, CountingTropicalF64; all=true)[]
    @test res.n == 5
    @test length(res.c.data) == 6
end

@testset "match polynomial" begin
    g = SimpleGraph(7)
    for (i,j) in [(1,2),(2,3),(3,4),(4,5),(5,6),(6,1),(1,7)]
        add_edge!(g, i, j)
    end
    @test graph_polynomial(Matching(g), Val(:polynomial))[] == Polynomial([1,7,13,5])
    g = smallgraph(:petersen)
    @test graph_polynomial(Matching(g), Val(:polynomial))[].coeffs == [6, 90, 145, 75, 15, 1][end:-1:1]
end

@testset "weighted matching" begin
    g = smallgraph(:petersen)
    problem = Matching(g; weights=fill(2, 15))
    @test solve(problem, SizeMax())[].n == 10
    res = solve(problem, SingleConfigMax())[].c.data
    @test is_matching(g, res)
    @test count_ones(res) * 2 == 10
end
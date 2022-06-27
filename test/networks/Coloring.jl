using Test, GenericTensorNetworks, Graphs

@testset "enumerating - coloring" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2),(2,3),(3,4),(4,1),(1,5),(2,4)]
        add_edge!(g, i, j)
    end
    code = Coloring{3}(g; optimizer=GreedyMethod())
    res = GenericTensorNetworks.best_solutions(code; all=true)[]
    @test length(res.c.data) == 12
    g = smallgraph(:petersen)
    code = Coloring{3}(g; optimizer=GreedyMethod())
    res = GenericTensorNetworks.best_solutions(code; all=true)[]
    @test length(res.c.data) == 120

    c = solve(code, SingleConfigMax())[]
    @test c.c.data ∈ res.c.data
    @test is_vertex_coloring(g, c.c.data)
end


@testset "weighted coloring" begin
    g = smallgraph(:petersen)
    problem = Coloring{3}(g; weights=fill(2, 15))
    @test weights(problem) == fill(2, 15)
    @test weights(chweights(problem, fill(3, 15))) == fill(3, 15)
    @test solve(problem, SizeMax())[].n == 30
    res = solve(problem, SingleConfigMax())[].c.data
    @test is_vertex_coloring(g, res)
end

@testset "empty graph" begin
    g = SimpleGraph(4)
    pb = Coloring{3}(g)
    @test solve(pb, SizeMax()) !== 4
end
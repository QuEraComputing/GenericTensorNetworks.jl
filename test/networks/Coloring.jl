using Test, GraphTensorNetworks, Graphs

@testset "enumerating - coloring" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2),(2,3),(3,4),(4,1),(1,5),(2,4)]
        add_edge!(g, i, j)
    end
    code = Coloring{3}(g; optimizer=GreedyMethod())
    res = best_solutions(code; all=true)[]
    @test length(res.c.data) == 12
    g = smallgraph(:petersen)
    code = Coloring{3}(g; optimizer=GreedyMethod())
    res = best_solutions(code; all=true)[]
    @test length(res.c.data) == 120

    c = solve(code, SingleConfigMax())[]
    @test c.c.data ∈ res.c.data
    @test is_good_vertex_coloring(g, c.c.data)
end


using Test, GenericTensorNetworks, Graphs

@testset "enumerating - coloring" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2),(2,3),(3,4),(4,1),(1,5),(2,4)]
        add_edge!(g, i, j)
    end
    code = GenericTensorNetwork(Coloring{3}(g); optimizer=GreedyMethod())
    res = GenericTensorNetworks.best_solutions(code; all=true)[]
    @test length(res.c.data) == 12
    g = smallgraph(:petersen)
    code = GenericTensorNetwork(Coloring{3}(g); optimizer=GreedyMethod())
    res = GenericTensorNetworks.best_solutions(code; all=true)[]
    @test length(res.c.data) == 120

    c = solve(code, SingleConfigMax())[]
    @test c.c.data ∈ res.c.data
    @test is_vertex_coloring(g, c.c.data)
end


@testset "weighted coloring" begin
    g = smallgraph(:petersen)
    problem = GenericTensorNetwork(Coloring{3}(g, fill(2, 15)))
    @test get_weights(problem) == fill(2, 15)
    @test get_weights(chweights(problem, fill(3, 15))) == fill(3, 15)
    @test solve(problem, SizeMax())[].n == 30
    res = solve(problem, SingleConfigMax())[].c.data
    @test is_vertex_coloring(g, res)
end

@testset "empty graph" begin
    g = SimpleGraph(4)
    pb = GenericTensorNetwork(Coloring{3}(g))
    @test solve(pb, SizeMax()) !== 4
end

@testset "planar gadget checking" begin
    function graph_crossing_gadget()
        edges = [
            (1,2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 1),
            (9, 10), (10, 11), (11, 12), (12, 9),
            (1, 9), (3, 10), (5, 11), (7, 12),
            (2, 10), (4, 11), (6, 12), (8, 9),
            (13, 9), (13, 10), (13, 11), (13, 12),
        ]
        g = SimpleGraph(13)
        for (i, j) in edges
            add_edge!(g, i, j)
        end
        return g
    end

    g = graph_crossing_gadget()
    problem = GenericTensorNetwork(Coloring{3}(g); openvertices=[3, 5])
    res = solve(problem, ConfigsMax())
    for i=1:3
        for ci in res[i,i].c
            @test ci[1] === ci[3] === ci[5] === ci[7] == i-1
        end
    end
    for (i, j) in [(1, 2), (1, 3), (2, 3), (2,1), (3, 1), (3, 2)]
        for ci in res[i,j].c
            @test ci[1] === ci[5] == j-1
            @test ci[3] === ci[7] == i-1
        end
    end
end
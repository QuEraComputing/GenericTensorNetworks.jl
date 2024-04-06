using GenericTensorNetworks, Test, Graphs

@testset "set packing" begin
    sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]]  # each set is a vertex
    gp = set_packing_network(sets; optimizer=GreedyMethod())
    @test get_weights(gp) == UnitWeight()
    @test get_weights(chweights(gp, fill(3, 5))) == fill(3,5)
    res = GenericTensorNetworks.best_solutions(gp; all=true)[]
    @test res.n == 2
    @test BitVector(Bool[0,0,1,1,0]) ∈ res.c.data
    @test BitVector(Bool[1,0,0,1,0]) ∈ res.c.data
    @test BitVector(Bool[0,1,1,0,0]) ∈ res.c.data
    @test all(x->is_set_packing(sets, x),res.c)
end
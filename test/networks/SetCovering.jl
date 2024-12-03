using GenericTensorNetworks, Test, Graphs

@testset "set covering" begin
    sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]]  # each set is a vertex
    gp = GenericTensorNetwork(SetCovering(sets); optimizer=GreedyMethod())
    @test get_weights(gp) == UnitWeight(length(sets))
    @test get_weights(set_weights(gp, fill(3, 5))) == fill(3,5)
    res = solve(gp, ConfigsMin())[]
    @test res.n == 3
    @test BitVector(Bool[1,0,1,1,0]) ∈ res.c.data
    @test BitVector(Bool[1,0,1,0,1]) ∈ res.c.data
    print(typeof(res.c.data)) 
    @test all(x->is_set_covering(gp.problem,x),res.c.data)
end
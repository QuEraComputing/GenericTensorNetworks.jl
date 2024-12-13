using GenericTensorNetworks, Test, Graphs

@testset "set packing" begin
    sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]]  # each set is a vertex
    gp = GenericTensorNetwork(SetPacking(sets); optimizer=GreedyMethod())
    @test GenericTensorNetworks.weights(gp) == UnitWeight(length(sets))
    @test GenericTensorNetworks.weights(set_weights(gp, fill(3, 5))) == fill(3,5)
    res = GenericTensorNetworks.largest_solutions(gp; all=true)[]
    @test res.n == 2
    @test BitVector(Bool[0,0,1,1,0]) ∈ res.c.data
    @test BitVector(Bool[1,0,0,1,0]) ∈ res.c.data
    @test BitVector(Bool[0,1,1,0,0]) ∈ res.c.data
    @test all(x->is_set_packing(gp.problem, x),res.c) # need GenericTensorNetworks: is_set_packing
end
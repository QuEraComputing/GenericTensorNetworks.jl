using GraphTensorNetworks, Test, Graphs

@testset "set covering" begin
    sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]]  # each set is a vertex
    gp = SetCovering(sets; optimizer=GreedyMethod())
    res = solve(gp, ConfigsMin())[]
    @test res.n == 3
    @test BitVector(Bool[1,0,1,1,0]) ∈ res.c.data
    @test BitVector(Bool[1,0,1,0,1]) ∈ res.c.data
    @test all(x->is_set_covering(sets, x),res.c)
end
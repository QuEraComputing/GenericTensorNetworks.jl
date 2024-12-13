using GenericTensorNetworks, Test, Graphs

@testset "dominating set basic" begin
    g = smallgraph(:petersen)
    mask = trues(10)
    mask[neighbors(g, 1)] .= false
    @test is_dominating_set(g, mask)
    mask[1] = false
    @test !is_dominating_set(g, mask)
end

@testset "dominating set v.s. maximal IS" begin
    g = smallgraph(:petersen)
    gp1 = GenericTensorNetwork(DominatingSet(g))
    @test GenericTensorNetworks.weights(gp1) == UnitWeight(nv(g))
    @test GenericTensorNetworks.weights(set_weights(gp1, fill(3, 10))) == fill(3, 10)
    @test solve(gp1, SizeMax())[].n == 10
    res1 = solve(gp1, ConfigsMin())[].c
    gp2 = GenericTensorNetwork(MaximalIS(g))
    res2 = solve(gp2, ConfigsMin())[].c
    @test res1 == res2
    configs = solve(gp2, ConfigsAll())[]
    for config in configs
        @test is_dominating_set(g, config)
    end
end

@testset "empty graph" begin
    g = SimpleGraph(4)
    pb = GenericTensorNetwork(DominatingSet(g))
    @test solve(pb, SizeMax()) !== 4
end
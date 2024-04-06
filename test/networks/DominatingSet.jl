using GenericTensorNetworks, Test, Graphs

@testset "dominating set basic" begin
    g = smallgraph(:petersen)
    mask = trues(10)
    mask[neighbors(g, 1)] .= false
    @test is_dominating_set(g, mask)
    mask[1] = false
    @test !is_dominating_set(g, mask)

    @test GenericTensorNetworks.dominating_set_tensor(TropicalF64(0), TropicalF64(1), 3)[:,:,1] == TropicalF64[-Inf 0.0; 0 0]
    @test GenericTensorNetworks.dominating_set_tensor(TropicalF64(0), TropicalF64(1), 3)[:,:,2] == TropicalF64[1.0 1.0; 1.0 1.0]
end

@testset "dominating set v.s. maximal IS" begin
    g = smallgraph(:petersen)
    gp1 = DominatingSet(g)
    @test get_weights(gp1) == UnitWeight()
    @test get_weights(chweights(gp1, fill(3, 10))) == fill(3, 10)
    @test solve(gp1, SizeMax())[].n == 10
    res1 = solve(gp1, ConfigsMin())[].c
    gp2 = MaximalIS(g)
    res2 = solve(gp2, ConfigsMin())[].c
    @test res1 == res2
    configs = solve(gp2, ConfigsAll())[]
    for config in configs
        @test is_dominating_set(g, config)
    end
end

@testset "empty graph" begin
    g = SimpleGraph(4)
    pb = DominatingSet(g)
    @test solve(pb, SizeMax()) !== 4
end
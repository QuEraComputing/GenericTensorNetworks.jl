using Test, GenericTensorNetworks, Graphs

@testset "deprecate" begin
    @test Independence(smallgraph(:petersen)) == IndependentSet(smallgraph(:petersen))
    @test MaximalIndependence(smallgraph(:petersen)) == MaximalIS(smallgraph(:petersen))
    @test_throws ErrorException UnitWeight()
    @test_throws ErrorException ZeroWeight()
    @test HyperSpinGlass(smallgraph(:petersen), ones(Int, ne(smallgraph(:petersen))), ones(Int, nv(smallgraph(:petersen)))) == SpinGlass(smallgraph(:petersen), ones(Int, ne(smallgraph(:petersen))), ones(Int, nv(smallgraph(:petersen))))

    idp = IndependentSet(smallgraph(:petersen))
    @test nflavor(idp) == num_flavors(idp)
    @test labels(idp) == variables(idp)
    @test get_weights(idp) == GenericTensorNetworks.weights(idp)
    @test chweights(idp, 2 * ones(Int, nv(idp.graph))) == set_weights(idp, 2 * ones(Int, nv(idp.graph)))

    @test GenericTensorNetworks.GraphProblem === ConstraintSatisfactionProblem
    sg = SpinGlass(smallgraph(:petersen), ones(Int, ne(smallgraph(:petersen))), ones(Int, nv(smallgraph(:petersen))))
    cfg = rand([0, 1], nv(sg.graph))
    @test spinglass_energy(sg, cfg) == energy(sg, cfg)
    @test unit_disk_graph([(1, 2), (2, 2)], 1.6) isa UnitDiskGraph
    @test solve(sg, SizeMax()) == solve(GenericTensorNetwork(sg), SizeMax())
end

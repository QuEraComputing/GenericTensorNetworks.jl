using GenericTensorNetworks, Graphs, Test

@testset "save load" begin
    M = 10
    fname = tempname()
    m = ConfigEnumerator([StaticBitVector(rand(Bool, 300)) for i=1:M])
    bm = GenericTensorNetworks.plain_matrix(m)
    rm = GenericTensorNetworks.raw_matrix(m)
    m1 = GenericTensorNetworks.from_raw_matrix(rm; bitlength=300, num_flavors=2)
    m2 = GenericTensorNetworks.from_plain_matrix(bm; num_flavors=2)
    @test m1 == m
    @test m2 == m
    save_configs(fname, m; format=:binary)
    @test_throws ErrorException load_configs("_test.bin"; format=:binary)
    ma = load_configs(fname; format=:binary, bitlength=300, num_flavors=2)
    @test ma == m

    fname = tempname()
    save_configs(fname, m; format=:text)
    mb = load_configs(fname; format=:text, num_flavors=2)
    @test mb == m

    M = 10
    m = ConfigEnumerator([StaticElementVector(3, rand(0:2, 300)) for i=1:M])
    bm = GenericTensorNetworks.plain_matrix(m)
    rm = GenericTensorNetworks.raw_matrix(m)
    m1 = GenericTensorNetworks.from_raw_matrix(rm; bitlength=300, num_flavors=3)
    m2 = GenericTensorNetworks.from_plain_matrix(bm; num_flavors=3)
    @test m1 == m
    @test m2 == m
    @test Matrix(m) == bm
    @test Vector(m.data[1]) == bm[:,1]

    fname = tempname()
    save_configs(fname, m; format=:binary)
    @test_throws ErrorException load_configs(fname; format=:binary)
    ma = load_configs(fname; format=:binary, bitlength=300, num_flavors=3)
    @test ma == m

    fname = tempname()
    save_configs(fname, m; format=:text)
    mb = load_configs(fname; format=:text, num_flavors=3)
    @test mb == m
end

@testset "save load tree" begin
    fname = tempname()
    tree = solve(GenericTensorNetwork(IndependentSet(smallgraph(:petersen))), ConfigsAll(; tree_storage=true))[]
    save_sumproduct(fname, tree)
    ma = load_sumproduct(fname)
    @test ma == tree
end

@testset "save load GenericTensorNetwork" begin
    g = smallgraph(:petersen)
    problem = IndependentSet(g, UnitWeight(10))
    tn = GenericTensorNetwork(problem; fixedvertices=Dict(1=>0, 2=>1))
    folder = tempname()
    GenericTensorNetworks.save_tensor_network(tn; folder=folder)
    tn2 = GenericTensorNetworks.load_tensor_network(folder)
    @test tn.problem == tn2.problem
    @test tn.code == tn2.code
    @test tn.fixedvertices == tn2.fixedvertices
    @test solve(tn, SizeMax()) == solve(tn2, SizeMax())

    # test with empty fixedvertices
    tn3 = GenericTensorNetwork(problem)
    folder2 = tempname()
    GenericTensorNetworks.save_tensor_network(tn3; folder=folder2)
    tn4 = GenericTensorNetworks.load_tensor_network(folder2)
    @test tn3.problem == tn4.problem
    @test tn3.code == tn4.code
    @test tn3.fixedvertices == tn4.fixedvertices

    # test error cases
    empty_folder = tempname()
    mkpath(empty_folder)
    @test_throws SystemError GenericTensorNetworks.load_tensor_network(empty_folder)
end


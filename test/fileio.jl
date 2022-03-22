using GraphTensorNetworks, Graphs, Test

@testset "save load" begin
    M = 10
    fname = tempname()
    m = ConfigEnumerator([StaticBitVector(rand(Bool, 300)) for i=1:M])
    bm = GraphTensorNetworks.plain_matrix(m)
    rm = GraphTensorNetworks.raw_matrix(m)
    m1 = GraphTensorNetworks.from_raw_matrix(rm; bitlength=300, nflavors=2)
    m2 = GraphTensorNetworks.from_plain_matrix(bm; nflavors=2)
    @test m1 == m
    @test m2 == m
    save_configs(fname, m; format=:binary)
    @test_throws ErrorException load_configs("_test.bin"; format=:binary)
    ma = load_configs(fname; format=:binary, bitlength=300, nflavors=2)
    @test ma == m

    fname = tempname()
    save_configs(fname, m; format=:text)
    mb = load_configs(fname; format=:text, nflavors=2)
    @test mb == m

    M = 10
    m = ConfigEnumerator([StaticElementVector(3, rand(0:2, 300)) for i=1:M])
    bm = GraphTensorNetworks.plain_matrix(m)
    rm = GraphTensorNetworks.raw_matrix(m)
    m1 = GraphTensorNetworks.from_raw_matrix(rm; bitlength=300, nflavors=3)
    m2 = GraphTensorNetworks.from_plain_matrix(bm; nflavors=3)
    @test m1 == m
    @test m2 == m
    @test Matrix(m) == bm
    @test Vector(m.data[1]) == bm[:,1]

    fname = tempname()
    save_configs(fname, m; format=:binary)
    @test_throws ErrorException load_configs(fname; format=:binary)
    ma = load_configs(fname; format=:binary, bitlength=300, nflavors=3)
    @test ma == m

    fname = tempname()
    save_configs(fname, m; format=:text)
    mb = load_configs(fname; format=:text, nflavors=3)
    @test mb == m
end

@testset "save load tree" begin
    fname = tempname()
    tree = solve(IndependentSet(smallgraph(:petersen)), ConfigsAll(; tree_storage=true))[]
    save_sumproduct(fname, tree)
    ma = load_sumproduct(fname)
    @test ma == tree
end


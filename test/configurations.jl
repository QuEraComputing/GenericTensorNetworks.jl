using GraphTensorNetworks, Test, LightGraphs
using OMEinsum
using TropicalNumbers: CountingTropicalF64

@testset "Config types" begin
    T = sampler_type(CountingTropical{Float32}, 5, 2)
    x = one(T)
    @test x.n === 0f0
    @test x.c.data == falses(5)
    x = zero(T)
    @test x.n === Float32(-Inf)
    @test x.c.data == trues(5)
    x = onehotv(T, 2, 1)
    @test x.n === 1f0
    @test x.c.data == Bool[0,1,0,0,0]

    T = set_type(CountingTropical{Float32}, 5, 2)
    x = one(T)
    @test x.n === 0f0
    @test length(x.c.data) == 1 && x.c.data[1] == falses(5)
    x = zero(T)
    @test x.n === Float32(-Inf)
    @test length(x.c.data) == 0
    x = onehotv(T, 2, 1)
    @test x.n === 1f0
    @test length(x.c.data) == 1 && x.c.data[1] == Bool[0,1,0,0,0]
    x = onehotv(T, 2, 0)
    @test x.c.data[1].data[1] == 0
end

@testset "enumerating" begin
    rawcode = Independence(random_regular_graph(10, 3); optmethod=:raw)
    optcode = Independence(optimize_code(rawcode.code; optmethod=:auto))
    for code in [rawcode, optcode]
        res0 = GraphTensorNetworks.mis_size(code)
        res1 = GraphTensorNetworks.mis_count(code)
        res2 = optimalsolutions(code; all=true)[]
        res3 = solutions(code, CountingTropical{Float64}; all=false)[]
        res4 = solutions(code, CountingTropical{Float64}; all=true)[]
        @test res0 == res2.n == res3.n == res4.n
        @test res1 == length(res2.c) == length(res4.c)
        @test res3.c.data ∈ res2.c.data
        @test res3.c.data ∈ res4.c.data
        res5 = optimalsolutions(code; all=false)[]
        @test res5.n == res0
        @test res5.c.data ∈ res2.c.data
    end
end

@testset "set packing" begin
    sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]]  # each set is a vertex
    gp = set_packing(sets; optmethod=:auto)
    res = optimalsolutions(gp; all=true)[]
    @test res.n == 2
    @test BitVector(Bool[0,0,1,1,0]) ∈ res.c.data
    @test BitVector(Bool[1,0,0,1,0]) ∈ res.c.data
    @test BitVector(Bool[0,1,1,0,0]) ∈ res.c.data
end

@testset "enumerating - max cut" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2),(2,3),(3,4),(4,1),(1,5),(2,4)]
        add_edge!(g, i, j)
    end
    code = MaxCut(g; optmethod=:greedy)
    res = optimalsolutions(code; all=true)[]
    @test length(res.c.data) == 2
    @test sum(res.c.data[1]) == 5
end

@testset "enumerating - coloring" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2),(2,3),(3,4),(4,1),(1,5),(2,4)]
        add_edge!(g, i, j)
    end
    code = Coloring{3}(g; optmethod=:greedy)
    res = solutions(code, CountingTropicalF64; all=true)[]
    @test length(res.c.data) == 12
    g = smallgraph(:petersen)
    code = Coloring{3}(g; optmethod=:greedy)
    res = solutions(code, CountingTropicalF64; all=true)[]
    @test length(res.c.data) == 120
end
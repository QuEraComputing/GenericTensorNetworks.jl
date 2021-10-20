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
    rawcode = Independence(random_regular_graph(10, 3); optimizer=nothing)
    optcode = Independence(optimize_code(rawcode.code, uniformsize(rawcode.code, 2), GreedyMethod()))
    for code in [rawcode, optcode]
        res0 = max_size(code)
        _, res1 = max_size_count(code)
        res2 = best_solutions(code; all=true)[]
        res3 = solutions(code, CountingTropical{Float64}; all=false)[]
        res4 = solutions(code, CountingTropical{Float64}; all=true)[]
        @test res0 == res2.n == res3.n == res4.n
        @test res1 == length(res2.c) == length(res4.c)
        @test res3.c.data ∈ res2.c.data
        @test res3.c.data ∈ res4.c.data
        res5 = best_solutions(code; all=false)[]
        @test res5.n == res0
        @test res5.c.data ∈ res2.c.data
        res6 = best2_solutions(code; all=true)[]
        res7 = all_solutions(code)[]
        idp = graph_polynomial(code, Val(:finitefield))[]
        @test all(x->x ∈ res7.coeffs[end-1].data, res6.a.data)
        @test all(x->x ∈ res7.coeffs[end].data, res6.b.data)
        for (i, (s, c)) in enumerate(zip(res7.coeffs, idp.coeffs))
            @test length(s) == c
            @test all(x->count_ones(x)==(i-1), s.data)
        end
    end
end

@testset "set packing" begin
    sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]]  # each set is a vertex
    gp = set_packing(sets; optimizer=GreedyMethod())
    res = best_solutions(gp; all=true)[]
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
    code = MaxCut(g; optimizer=GreedyMethod())
    res = best_solutions(code; all=true)[]
    @test length(res.c.data) == 2
    @test sum(res.c.data[1]) == 5
end

@testset "enumerating - coloring" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2),(2,3),(3,4),(4,1),(1,5),(2,4)]
        add_edge!(g, i, j)
    end
    code = Coloring{3}(g; optimizer=GreedyMethod())
    res = solutions(code, CountingTropicalF64; all=true)[]
    @test length(res.c.data) == 12
    g = smallgraph(:petersen)
    code = Coloring{3}(g; optimizer=GreedyMethod())
    res = solutions(code, CountingTropicalF64; all=true)[]
    @test length(res.c.data) == 120
end

@testset "enumerating - matching" begin
    g = smallgraph(:petersen)
    code = Matching(g; optimizer=GreedyMethod())
    res = solutions(code, CountingTropicalF64; all=true)[]
    @test res.n == 5
    @test length(res.c.data) == 6
end
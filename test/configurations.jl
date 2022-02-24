using GraphTensorNetworks, Test, Graphs
using OMEinsum
using TropicalNumbers: CountingTropicalF64
using OMEinsumContractionOrders: uniformsize
using GraphTensorNetworks: _onehotv, _x, sampler_type, set_type

@testset "Config types" begin
    T = sampler_type(CountingTropical{Float32}, 5, 2)
    x = one(T)
    @test x.n === 0f0
    @test x.c.data == falses(5)
    x = zero(T)
    @test x.n === Float32(-Inf)
    @test x.c.data == trues(5)
    x = _onehotv(T, 2, 1)
    @test x.n === 0f0
    @test x.c.data == Bool[0,1,0,0,0]

    T = set_type(CountingTropical{Float32}, 5, 2)
    x = one(T)
    @test x.n === 0f0
    @test length(x.c.data) == 1 && x.c.data[1] == falses(5)
    x = zero(T)
    @test x.n === Float32(-Inf)
    @test length(x.c.data) == 0
    x = _onehotv(T, 2, 1)
    @test x.n === 0f0
    @test length(x.c.data) == 1 && x.c.data[1] == Bool[0,1,0,0,0]
    x = _onehotv(T, 2, 0)
    @test x.c.data[1].data[1] == 0
end

@testset "enumerating" begin
    rawcode = IndependentSet(smallgraph(:petersen); optimizer=nothing)
    optcode = IndependentSet(optimize_code(rawcode.code, uniformsize(rawcode.code, 2), GreedyMethod()), 10, UnWeighted())
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
        res6_ = bestk_solutions(code, 2)[]
        res7 = all_solutions(code)[]
        idp = graph_polynomial(code, Val(:finitefield))[]
        @test all(x->x ∈ res7.coeffs[end-1].data, res6.coeffs[1].data)
        @test all(x->x ∈ res7.coeffs[end].data, res6.coeffs[2].data)
        @test all(x->x ∈ res7.coeffs[end-1].data, res6_.coeffs[1].data)
        @test all(x->x ∈ res7.coeffs[end].data, res6_.coeffs[2].data)
        for (i, (s, c)) in enumerate(zip(res7.coeffs, idp.coeffs))
            @test length(s) == c
            @test all(x->count_ones(x)==(i-1), s.data)
        end
    end
end
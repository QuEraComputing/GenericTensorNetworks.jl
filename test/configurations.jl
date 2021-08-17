using GraphTensorNetworks, Test, LightGraphs
using OMEinsum

@testset "ConfigTropical" begin
    x = one(ConfigTropical{Float64, 5, 1})
    @test x.n == 0
    @test x.config == falses(5)
    x = zero(ConfigTropical{Float64, 5, 1})
    @test x.n == -Inf
    @test x.config == trues(5)
end

@testset "enumerating" begin
    rawcode = Independence(random_regular_graph(10, 3))
    optcode = optimize_code(rawcode; method=:kahypar)
    for code in [rawcode, optcode]
        res0 = GraphTensorNetworks.mis_size(code)
        res1 = GraphTensorNetworks.mis_count(code)
        res2 = getconfigs_bounded(code; all=true)[]
        res3 = getconfigs_direct(code; all=false)[]
        res4 = getconfigs_direct(code; all=true)[]
        @test res0 == res2.n == res3.n == res4.n
        @test res1 == length(res2.c) == length(res4.c)
        @test res3.c.data ∈ res2.c.data
        @test res3.c.data ∈ res4.c.data
        res5 = getconfigs_bounded(code; all=false)[]
        @test res5.n == res0
        @test res5.c.data ∈ res2.c.data
    end
end
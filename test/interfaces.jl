using GenericTensorNetworks
using Graphs, Test, Random

@testset "independent set problem" begin
    g = Graphs.smallgraph("petersen")
    for optimizer in (GreedyMethod(), TreeSA(ntrials=1))
        Random.seed!(3)
        gp = IndependentSet(g; optimizer=optimizer)
        @test contraction_complexity(gp).sc == 4
        @test timespacereadwrite_complexity(gp)[2] == 4
        @test timespace_complexity(gp)[2] == 4
        res1 = solve(gp, SizeMax())[]
        res2 = solve(gp, CountingAll())[]
        res3 = solve(gp, CountingMax(Single))[]
        res4 = solve(gp, CountingMax(2))[]
        res5 = solve(gp, GraphPolynomial(; method = :polynomial))[]
        res6 = solve(gp, SingleConfigMax())[]
        res7 = solve(gp, ConfigsMax(;bounded=false))[]
        res8 = solve(gp, ConfigsMax(2; bounded=false))[]
        res9 = solve(gp, ConfigsAll())[]
        res10 = solve(gp, GraphPolynomial(method=:fft))[]
        res11 = solve(gp, GraphPolynomial(method=:finitefield))[]
        res12 = solve(gp, SingleConfigMax(; bounded=true))[]
        res13 = solve(gp, ConfigsMax(;bounded=true))[]
        res14 = solve(gp, CountingMax(3))[]
        res15 = solve(gp, ConfigsMax(3))[]
        res16 = solve(gp, ConfigsMax(2; bounded=true))[]
        res17 = solve(gp, ConfigsMax(3; bounded=true))[]
        res18 = solve(gp, PartitionFunction(0.0))[]
        @test res1.n == 4
        @test res2 == 76
        @test res3.n == 4 && res3.c == 5
        @test res4.maxorder == 4 && res4.coeffs[1] == 30 && res4.coeffs[2]==5
        @test res5 == Polynomial([1.0, 10.0, 30, 30, 5])
        @test res6.c.data ∈ res7.c.data
        @test all(x->sum(x) == 4, res7.c.data)
        @test all(x->sum(x) == 3, res8.coeffs[1].data) && all(x->sum(x) == 4, res8.coeffs[2].data) && length(res8.coeffs[1].data) == 30 && length(res8.coeffs[2].data) == 5
        @test length(unique(res9.data)) == 76 && all(c->is_independent_set(g, c), res9.data)
        @test res10 ≈ res5
        @test res11 == res5
        @test res12.c.data ∈ res13.c.data
        @test res13.c == res7.c
        @test res14.maxorder == 4 && res14.coeffs[1]==30 && res14.coeffs[2] == 30 && res14.coeffs[3]==5
        @test all(x->sum(x) == 2, res15.coeffs[1].data) && all(x->sum(x) == 3, res15.coeffs[2].data) && all(x->sum(x) == 4, res15.coeffs[3].data) &&
                length(res15.coeffs[1].data) == 30 && length(res15.coeffs[2].data) == 30 && length(res15.coeffs[3].data) == 5
        @test all(x->sum(x) == 3, res16.coeffs[1].data) && all(x->sum(x) == 4, res16.coeffs[2].data) && length(res16.coeffs[1].data) == 30 && length(res16.coeffs[2].data) == 5
        @test all(x->sum(x) == 2, res17.coeffs[1].data) && all(x->sum(x) == 3, res17.coeffs[2].data) && all(x->sum(x) == 4, res17.coeffs[3].data) &&
                length(res17.coeffs[1].data) == 30 && length(res17.coeffs[2].data) == 30 && length(res17.coeffs[3].data) == 5

        @test res18 ≈ res2  
    end
end

@testset "slicing" begin
    g = Graphs.smallgraph("petersen")
    gp = IndependentSet(g; optimizer=TreeSA(nslices=5, ntrials=1))
    res1 = solve(gp, SizeMax())[]
    res2 = solve(gp, CountingAll())[]
    res3 = solve(gp, CountingMax(Single))[]
    res4 = solve(gp, CountingMax(2))[]
    res5 = solve(gp, GraphPolynomial(; method = :polynomial))[]
    res6 = solve(gp, SingleConfigMax())[]
    res7 = solve(gp, ConfigsMax(;bounded=false))[]
    res8 = solve(gp, ConfigsMax(2; bounded=false))[]
    res9 = solve(gp, ConfigsAll())[]
    res10 = solve(gp, GraphPolynomial(method=:fft))[]
    res11 = solve(gp, GraphPolynomial(method=:finitefield))[]
    res12 = solve(gp, SingleConfigMax(; bounded=true))[]
    res13 = solve(gp, ConfigsMax(;bounded=true))[]
    res14 = solve(gp, CountingMax(3))[]
    res15 = solve(gp, ConfigsMax(3))[]
    res16 = solve(gp, ConfigsMax(2; bounded=true))[]
    res17 = solve(gp, ConfigsMax(3; bounded=true))[]
    @test res1.n == 4
    @test res2 == 76
    @test res3.n == 4 && res3.c == 5
    @test res4.maxorder == 4 && res4.coeffs[1] == 30 && res4.coeffs[2]==5
    @test res5 == Polynomial([1.0, 10.0, 30, 30, 5])
    @test res6.c.data ∈ res7.c.data
    @test all(x->sum(x) == 4, res7.c.data)
    @test all(x->sum(x) == 3, res8.coeffs[1].data) && all(x->sum(x) == 4, res8.coeffs[2].data) && length(res8.coeffs[1].data) == 30 && length(res8.coeffs[2].data) == 5
    @test length(unique(res9.data)) == 76 && all(c->is_independent_set(g, c), res9.data)
    @test res10 ≈ res5
    @test res11 == res5
    @test res12.c.data ∈ res13.c.data
    @test res13.c == res7.c
    @test res14.maxorder == 4 && res14.coeffs[1]==30 && res14.coeffs[2] == 30 && res14.coeffs[3]==5
    @test all(x->sum(x) == 2, res15.coeffs[1].data) && all(x->sum(x) == 3, res15.coeffs[2].data) && all(x->sum(x) == 4, res15.coeffs[3].data) &&
            length(res15.coeffs[1].data) == 30 && length(res15.coeffs[2].data) == 30 && length(res15.coeffs[3].data) == 5
    @test all(x->sum(x) == 3, res16.coeffs[1].data) && all(x->sum(x) == 4, res16.coeffs[2].data) && length(res16.coeffs[1].data) == 30 && length(res16.coeffs[2].data) == 5
    @test all(x->sum(x) == 2, res17.coeffs[1].data) && all(x->sum(x) == 3, res17.coeffs[2].data) && all(x->sum(x) == 4, res17.coeffs[3].data) &&
            length(res17.coeffs[1].data) == 30 && length(res17.coeffs[2].data) == 30 && length(res17.coeffs[3].data) == 5
end

@testset "Weighted" begin
    g = Graphs.smallgraph("petersen")
    gp = IndependentSet(g; weights=collect(1:10))
    res1 = solve(gp, SizeMax())[]
    @test res1 == Tropical(24.0)
    res2 = solve(gp, CountingMax(Single))[]
    @test res2 == CountingTropical(24.0, 1.0)
    res2b = solve(gp, CountingMax(2))[]
    @test res2b == Max2Poly(1.0, 1.0, 24.0)
    res3 = solve(gp, SingleConfigMax(; bounded=false))[]
    res3b = solve(gp, SingleConfigMax(; bounded=true))[]
    @test res3 == res3b
    @test sum(collect(res3.c.data) .* (1:10))  == 24.0
    res4 = solve(gp, ConfigsMax(Single; bounded=false))[]
    res4b = solve(gp, ConfigsMax(Single; bounded=true))[]
    @test res4 == res4b
    @test res4.c.data[] == res3.c.data

    g = Graphs.smallgraph("petersen")
    gp = IndependentSet(g; weights=fill(0.5, 10))
    res5 = solve(gp, SizeMax(6))[]
    @test res5.orders == Tropical.([3.0,4,4,4,4,4] ./ 2)
    res6 = solve(gp, SingleConfigMax(6))[]
    @test all(enumerate(res6.orders)) do r
        i, o = r
        is_independent_set(g, o.c.data) && count_ones(o.c.data) == (i==1 ? 3 : 4)
    end
end

@testset "tree storage" begin
    g = smallgraph(:petersen)
    gp = IndependentSet(g)
    res1 = solve(gp, ConfigsAll(; tree_storage=true))[]
    res2 = solve(gp, ConfigsAll(; tree_storage=false))[]
    @test res1 isa SumProductTree && res2 isa ConfigEnumerator
    @test length(res1) == length(res2)
    @test Set(res2 |> collect) == Set(res1 |> collect)

    res1 = solve(gp, ConfigsMax(; tree_storage=true))[].c
    res2 = solve(gp, ConfigsMax(; tree_storage=false))[].c
    @test res1 isa SumProductTree && res2 isa ConfigEnumerator
    @test length(res1) == length(res2)
    @test Set(res2 |> collect) == Set(res1 |> collect)

    res1s = solve(gp, ConfigsMax(2; tree_storage=true))[].coeffs
    res2s = solve(gp, ConfigsMax(2; tree_storage=false))[].coeffs
    for (res1, res2) in zip(res1s, res2s)
        @test res1 isa SumProductTree && res2 isa ConfigEnumerator
        @test length(res1) == length(res2)
        @test Set(res2 |> collect) == Set(res1 |> collect)
    end

    res1 = solve(gp, ConfigsMin(; tree_storage=true))[].c
    res2 = solve(gp, ConfigsMin(; tree_storage=false))[].c
    @test res1 isa SumProductTree && res2 isa ConfigEnumerator
    @test length(res1) == length(res2)
    @test Set(res2 |> collect) == Set(res1 |> collect)

    res1s = solve(gp, ConfigsMin(2; tree_storage=true))[].coeffs
    res2s = solve(gp, ConfigsMin(2; tree_storage=false))[].coeffs
    for (res1, res2) in zip(res1s, res2s)
        @test res1 isa SumProductTree && res2 isa ConfigEnumerator
        @test length(res1) == length(res2)
        @test Set(res2 |> collect) == Set(res1 |> collect)
    end
end

@testset "memory estimation" begin
    gp = IndependentSet(smallgraph(:petersen))
    for property in [
            SizeMax(), SizeMin(), SizeMax(3), SizeMin(3), CountingMax(), CountingMin(), CountingMax(2), CountingMin(2),
            ConfigsMax(;bounded=true), ConfigsMin(;bounded=true), ConfigsMax(2;bounded=true), ConfigsMin(2;bounded=true), 
            ConfigsMax(;bounded=false), ConfigsMin(;bounded=false), ConfigsMax(2;bounded=false), ConfigsMin(2;bounded=false), SingleConfigMax(;bounded=false), SingleConfigMin(;bounded=false),
            CountingAll(), ConfigsAll(), SingleConfigMax(2), SingleConfigMin(2), SingleConfigMax(2; bounded=true), SingleConfigMin(2,bounded=true),
        ]
        @show property
        ET = GenericTensorNetworks.tensor_element_type(Float32, 10, 2, property)
        @test eltype(solve(gp, property, T=Float32)) <: ET
        @test estimate_memory(gp, property) isa Integer
    end
    @test GenericTensorNetworks.tensor_element_type(Float32, 10, 2, GraphPolynomial(method=:polynomial)) == Polynomial{Float32, :x}
    @test GenericTensorNetworks.tensor_element_type(Float32, 10, 2, GraphPolynomial(method=:laurent)) == LaurentPolynomial{Float32, :x}
    @test sizeof(GenericTensorNetworks.tensor_element_type(Float32, 10, 2, GraphPolynomial(method=:fitting))) == 4
    @test sizeof(GenericTensorNetworks.tensor_element_type(Float32, 10, 2, GraphPolynomial(method=:fft))) == 8
    @test sizeof(GenericTensorNetworks.tensor_element_type(Float64, 10, 2, GraphPolynomial(method=:finitefield))) == 4
    @test GenericTensorNetworks.tensor_element_type(Float32, 10, 2, SingleConfigMax(;bounded=true)) == Tropical{Float32}
    @test GenericTensorNetworks.tensor_element_type(Float32, 10, 2, SingleConfigMin(;bounded=true)) == Tropical{Float32}

    @test estimate_memory(gp, SizeMax()) * 2 == estimate_memory(gp, CountingMax())
    @test estimate_memory(gp, SizeMax()) == estimate_memory(gp, PartitionFunction(1.0))
    @test estimate_memory(gp, SingleConfigMax(bounded=true)) > estimate_memory(gp, SingleConfigMax(bounded=false))
    @test estimate_memory(gp, ConfigsMax(bounded=true)) == estimate_memory(gp, SingleConfigMax(bounded=false))
    @test estimate_memory(gp, GraphPolynomial(method=:fitting); T=Float32) * 4 == estimate_memory(gp, GraphPolynomial(method=:fft))
    @test estimate_memory(gp, GraphPolynomial(method=:finitefield)) * 10 == estimate_memory(gp, GraphPolynomial(method=:polynomial); T=Float32)
end

@testset "error on not solvable problems" begin
    gp = IndependentSet(smallgraph(:petersen); weights=rand(1:3, 10))
    @test !GenericTensorNetworks.has_noninteger_weights(gp)
    gp = IndependentSet(smallgraph(:petersen); weights=rand(10))
    @test GenericTensorNetworks.has_noninteger_weights(gp)
    @test_throws ArgumentError solve(gp, GraphPolynomial())
    @test_throws ArgumentError solve(gp, CountingMax(2))
    @test solve(gp, CountingMax(Single)) isa Array
    @test_throws ArgumentError solve(gp, ConfigsMax(2))
    @test solve(gp, CountingMin(Single)) isa Array
    @test_throws ArgumentError solve(gp, ConfigsMin(2))
    @test solve(gp, ConfigsMax(Single)) isa Array
    @test_throws ArgumentError solve(gp, ConfigsMin(2))
    @test solve(gp, ConfigsMin(Single)) isa Array
    @test_throws AssertionError ConfigsMin(0)
end
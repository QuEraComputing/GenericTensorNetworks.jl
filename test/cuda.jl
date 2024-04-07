using CUDA, Random
using LinearAlgebra: mul!
using GenericTensorNetworks, Test
using Graphs
CUDA.allowscalar(false)

@testset "cuda patch" begin
    for T in [Tropical{Float64}, CountingTropical{Float64,Float64}]
        a = T.(CUDA.randn(4, 4))
        b = T.(CUDA.randn(4))
        for A in [transpose(a), a, transpose(b)]
            for B in [transpose(a), a, b]
                if !(size(A) == (1,4) && size(B) == (4,))
                    res0 = Array(A) * Array(B)
                    res1 = A * B
                    res2 = mul!(CUDA.zeros(T, size(res0)...), A, B, true, false)
                    @test Array(res1) ≈ res0
                    @test Array(res2) ≈ res0
                end
            end
        end
    end
end

@testset "cuda functions" begin
    g = Graphs.smallgraph("petersen")
    item(x::AbstractArray) = Array(x)[]
    #optimizer = TreeSA(ntrials=1)
    optimizer = GreedyMethod()
    gp = GenericTensorNetwork(IndependentSet(g); optimizer=optimizer)
    @test contraction_complexity(gp).sc == 4
    @test timespacereadwrite_complexity(gp)[2] == 4
    @test timespace_complexity(gp)[2] == 4
    res1 = solve(gp, SizeMax(); usecuda=true) |> item
    res2 = solve(gp, CountingAll(); usecuda=true) |> item
    res3 = solve(gp, CountingMax(Single); usecuda=true) |> item
    res4 = solve(gp, CountingMax(2); usecuda=true) |> item
    res5 = solve(gp, GraphPolynomial(; method = :polynomial))[]
    res6 = solve(gp, SingleConfigMax(); usecuda=true) |> item
    res7 = solve(gp, ConfigsMax(;bounded=false))[]
    res10 = solve(gp, GraphPolynomial(method=:fft); usecuda=true) |> item
    res11 = solve(gp, GraphPolynomial(method=:finitefield); usecuda=true) |> item
    res12 = solve(gp, SingleConfigMax(; bounded=true); usecuda=true) |> item
    res13 = solve(gp, ConfigsMax(;bounded=true))[]
    res14 = solve(gp, CountingMax(3); usecuda=true) |> item
    res18 = solve(gp, PartitionFunction(0.0); usecuda=true) |> item
    @test res1.n == 4
    @test res2 == 76
    @test res3.n == 4 && res3.c == 5
    @test res4.maxorder == 4 && res4.coeffs[1] == 30 && res4.coeffs[2]==5
    @test res6.c.data ∈ res7.c.data
    @test res10 ≈ res5
    @test res11 == res5
    @test res12.c.data ∈ res13.c.data
    @test res14.maxorder == 4 && res14.coeffs[1]==30 && res14.coeffs[2] == 30 && res14.coeffs[3]==5
    @test res18 ≈ res2  
end

@testset "spinglass" begin
    g = Graphs.smallgraph("petersen")
    gp = GenericTensorNetwork(SpinGlass(g, UnitWeight()))
    usecuda=true
    @test solve(gp, CountingMax(); usecuda) isa CuArray
    gp2 = GenericTensorNetwork(SpinGlass(g, UnitWeight()); openvertices=(2,))
    @test solve(gp2, CountingMax(); usecuda) isa CuArray
end

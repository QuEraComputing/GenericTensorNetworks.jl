using GenericTensorNetworks: select_dims
using Test

@testset "select dims" begin
    a, b, c = randn(2), randn(2, 2), randn(2,2,2)
    a_, b_, c_ = select_dims([a, b, c], [[1], [1,2], [1,2,3]], Dict(1=>1, 3=>0))
    @test a_ == a[2:2]
    @test b_ == b[2:2, :]
    @test c_ == c[2:2, :, 1:1]
    a, b = randn(3), randn(3,3,3)
    a_, b_ = select_dims([a, b], [[1], [2,3,1]], Dict(1=>1, 3=>2))
    @test a_ == a[2:2]
    @test b_ == b[:,3:3,2:2]
end

include("IndependentSet.jl")
include("MaximalIS.jl")
include("MaxCut.jl")
include("PaintShop.jl")
include("Coloring.jl")
include("Matching.jl")
include("Satisfiability.jl")
include("DominatingSet.jl")
include("SetPacking.jl")
include("SetCovering.jl")
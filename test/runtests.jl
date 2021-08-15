using GraphTensorNetworks
using Test

@testset "independence polynomial" begin
    include("independence_polynomial.jl")
end

@testset "configurations" begin
    include("configurations.jl")
end

@testset "bounding" begin
    include("bounding.jl")
end
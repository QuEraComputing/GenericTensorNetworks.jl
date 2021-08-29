using GraphTensorNetworks
using Test

@testset "bitvector" begin
    include("bitvector.jl")
end

@testset "independence polynomial" begin
    include("graph_polynomials.jl")
end

@testset "configurations" begin
    include("configurations.jl")
end

@testset "bounding" begin
    include("bounding.jl")
end

@testset "interfaces" begin
    include("interfaces.jl")
end
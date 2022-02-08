using GraphTensorNetworks
using Test, Documenter

@testset "bitvector" begin
    include("bitvector.jl")
end

@testset "arithematics" begin
    include("arithematics.jl")
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

@testset "visualize" begin
    include("visualize.jl")
end

# --------------
# doctests
# --------------
doctest(GraphTensorNetworks)

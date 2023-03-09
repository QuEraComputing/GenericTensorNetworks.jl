using GenericTensorNetworks
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

@testset "networks" begin
    include("networks/networks.jl")
end

@testset "graphs" begin
    include("graphs.jl")
end

@testset "visualize" begin
    include("visualize.jl")
end

@testset "fileio" begin
    include("fileio.jl")
end

@testset "multiprocessing" begin
    include("multiprocessing.jl")
end

using CUDA
if CUDA.functional()
    @testset "cuda" begin
        include("cuda.jl")
    end
end

# --------------
# doctests
# --------------
doctest(GenericTensorNetworks)

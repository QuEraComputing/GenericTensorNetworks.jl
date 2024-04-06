using Distributed, GenericTensorNetworks.SimpleMultiprocessing
using Random, GenericTensorNetworks  # to avoid multi-precompilation
@everywhere using Random, GenericTensorNetworks

results = multiprocess_run(collect(1:10)) do seed
    Random.seed!(seed)
    n = 10
    @info "Graph size $n x $n, seed= $seed"
    g = random_diagonal_coupled_graph(n, n, 0.8)
    gp = GenericTensorNetwork(IndependentSet(g); optimizer=TreeSA())
    res = solve(gp, GraphPolynomial())[]
    return res
end

println(results)

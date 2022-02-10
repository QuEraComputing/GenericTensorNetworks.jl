# Performance Tips

## Optimize tensor network contraction order
```julia
julia> using GraphTensorNetworks, Graphs, Random

julia> graph = random_regular_graph(120, 3)
{120, 180} undirected simple Int64 graph

julia> problem = IndependentSet(graph; optimizer=TreeSA(
    sc_target=20, sc_weight=1.0, rw_weight=3.0, ntrials=10, βs=0.01:0.1:15.0, niters=20), simplifier=MergeGreedy());
```

Key word argument `optimizer` decides the contraction order optimizer of the tensor network.
Here, we choose the `TreeSA` optimizer to optimize the tensor network contraciton tree, it is a local search based algorithm.
It is one of the state of the art tensor network contraction order optimizers, one may check [arXiv: 2108.05665](https://arxiv.org/abs/2108.05665) to learn more about the algorithm.
Other optimizers include
* [`GreedyMethod`](@ref) (default, fastest in searching speed but worse in contraction order)
* [`TreeSA`](@ref)
* [`KaHyParBipartite`](@ref)
* [`SABipartite`](@ref)

One can type `?TreeSA` in a Julia REPL for more information about how to configure the hyper-parameters of `TreeSA` method.
`simplifier` keyword argument is not so important, it is a preprocessing routine to improve the searching speed of the `optimizer`.

The returned instance `problem` contains a field `code` that specifies the tensor network contraction order. For an independent set problem, its contraction time space complexity is ``2^{{\rm tw}(G)}``, where ``{\rm tw(G)}`` is the [tree-width](https://en.wikipedia.org/wiki/Treewidth) of ``G``.
One can check the time, space and read-write complexity with the following function.

```julia
julia> timespacereadwrite_complexity(problem)
```

The return values are `log2` of the the number of iterations, the number elements in the max tensor and the number of read-write operations to tensor elements.

## GEMM for Tropical numbers
You can speed up the Tropical number matrix multiplication when computing `SizeMax()` by using the Tropical GEMM routines implemented in package [`TropicalGEMM.jl`](https://github.com/TensorBFS/TropicalGEMM.jl/).

```julia
julia> using BenchmarkTools

julia> @btime solve(problem, SizeMax())
  91.630 ms (19203 allocations: 23.72 MiB)
0-dimensional Array{TropicalF64, 0}:
53.0ₜ

julia> using TropicalGEMM

julia> @btime solve(problem, SizeMax())
  8.960 ms (18532 allocations: 17.01 MiB)
0-dimensional Array{TropicalF64, 0}:
53.0ₜ
```

The `TropicalGEMM` pirates the `LinearAlgebra.mul!` interface, hence it takes effect upon using.
The GEMM routine can speed up the computation on CPU for one order, with multi-threading, it can be even faster.
Benchmark shows the performance of `TropicalGEMM` is close to the theoretical optimal value.

## Make use of GPUs
To upload the computing to GPU, you just add need to use CUDA, and offer a new key word argument.
```julia
julia> using CUDA
[ Info: OMEinsum loaded the CUDA module successfully

julia> solve(problem, SizeMax(), usecuda=true)
0-dimensional CuArray{TropicalF64, 0, CUDA.Mem.DeviceBuffer}:
53.0ₜ
```

CUDA backended properties are
* [`SizeMax`](@ref)
* [`CountingAll`](@ref)
* [`CountingMax`](@ref)
* [`GraphPolynomial`](@ref)
* [`SingleConfigMax`](@ref)
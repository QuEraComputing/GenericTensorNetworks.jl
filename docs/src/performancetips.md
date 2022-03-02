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
(21.90683335864693, 17.0, 20.03588509836998)
```

The return values are `log2` of the the number of iterations, the number elements in the largest tensor during contraction and the number of read-write operations to tensor elements.
In this example, the number of `+` and `*` operations are both `\sim 2^{21.9}`
and the number of read-write operations are `\sim 2^{20}`.
The largest tensor size is ``2^17``, one can check the element size by typing
```julia
julia> sizeof(TropicalF64)
8

julia> sizeof(TropicalF32)
4

julia> sizeof(StaticBitVector{200,4})
32

julia> sizeof(TruncatedPoly{5,Float64,Float64})
48
```

One can use [`estimate_memory`](@ref) to get a good estimation of peak memory in bytes.
```julia
julia> estimate_memory(problem, GraphPolynomial(; method=:finitefield))
297616

julia> estimate_memory(problem, GraphPolynomial(; method=:polynomial))
71427840
```
It means one only needs 298 KB memory to find the graph polynomial with the finite field approach,
but needs 71 MB memory to find the graph polynomial using the [`Polynomial`](@ref) type.

!!! note
    * The actual run time memory can be several times larger than the size of the maximum tensor.
    There is no constant bound for the factor, an empirical value for it is 3x.
    * For mutable types like [`Polynomial`](@ref) and [`ConfigEnumerator`](@ref), the `sizeof` function does not measure the actual element size.

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

## Sum product representation for configurations
[`TreeConfigEnumerator`](@ref) can save a lot memory for you to store exponential number of configurations in polynomial space.
It is a sum-product expression tree to store [`ConfigEnumerator`](@ref) in a lazy style, configurations can be extracted by depth first searching the tree with the `Base.collect` method. Although it is space efficient, it is in general not easy to extract information from it.
This tree structure supports directed sampling so that one can get some statistic properties from it with an intermediate effort.

(To be written.)

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
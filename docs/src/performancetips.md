# Performance Tips

## Optimize tensor network contraction orders
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
* [`GreedyMethod`](@ref) (default, fastest in searching speed but worst in contraction complexity)
* [`TreeSA`](@ref) (often best in contraction complexity, supports slicing)
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
In this example, the number of `+` and `*` operations are both ``\sim 2^{21.9}``
and the number of read-write operations are ``\sim 2^{20}``.
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
but needs 71 MB memory to find the graph polynomial using the [`Polynomial`](https://juliamath.github.io/Polynomials.jl/stable/polynomials/polynomial/#Polynomial-2) type.

!!! note
    * The actual run time memory can be several times larger than the size of the maximum tensor.
    There is no constant bound for the factor, an empirical value for it is 3x.
    * For mutable types like [`Polynomial`](https://juliamath.github.io/Polynomials.jl/stable/polynomials/polynomial/#Polynomial-2) and [`ConfigEnumerator`](@ref), the `sizeof` function does not measure the actual element size.

## Slicing

For large scale applications, it is also possible to slice over certain degrees of freedom to reduce the space complexity, i.e.
loop and accumulate over certain degrees of freedom so that one can have a smaller tensor network inside the loop due to the removal of these degrees of freedom.
In the [`TreeSA`](@ref) optimizer, one can set `nslices` to a value larger than zero to turn on this feature.

```julia
julia> using GraphTensorNetworks, Graphs, Random

julia> graph = random_regular_graph(120, 3)
{120, 180} undirected simple Int64 graph

julia> problem = IndependentSet(graph; optimizer=TreeSA(βs=0.01:0.1:25.0, ntrials=10, niters=10));

julia> timespacereadwrite_complexity(problem)
(20.856518235241687, 16.0, 18.88208476145812)

julia> problem = IndependentSet(graph; optimizer=TreeSA(βs=0.01:0.1:25.0, ntrials=10, niters=10, nslices=5));

julia> timespacereadwrite_complexity(problem)
(21.134967710592804, 11.0, 19.84529401927876)
```

In the second `IndependentSet` constructor, we slice over 5 degrees of freedom, which can reduce the space complexity by at most 5.
In this application, the slicing achieves the largest possible space complexity reduction 5, while the time and read-write complexity are only increased by less than 1,
i.e. the peak memory usage is reduced by a factor ``32``, while the (theoretical) computing time is increased by at a factor ``< 2``.

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
[`SumProductTree`](@ref) (an alias of [`SumProductTree`](@ref) with [`StaticElementVector`](@ref) as its data type) can save a lot memory for you to store exponential number of configurations in polynomial space.
It is a sum-product expression tree to store [`ConfigEnumerator`](@ref) in a lazy style, configurations can be extracted by depth first searching the tree with the `Base.collect` method. Although it is space efficient, it is in general not easy to extract information from it.
This tree structure supports directed sampling so that one can get some statistic properties from it with an intermediate effort.

For example, if we want to check some property of an intermediate scale graph, one can type
```julia
julia> graph = random_regular_graph(70, 3)

julia> problem = IndependentSet(graph; optimizer=TreeSA());

julia> tree = solve(problem, ConfigsAll(; tree_storage=true))[];
16633909006371
```
If one wants to store these configurations, he will need a hard disk of size 256 TB!
However, this sum-product binary tree structure supports efficient and unbiased direct sampling.

```julia
samples = generate_samples(tree, 1000);
```

With these samples, one can already compute useful properties like distribution of hamming distance (see [`hamming_distribution`](@ref)).

```julia
julia> using UnicodePlots

julia> lineplot(hamming_distribution(samples, samples))
          ┌────────────────────────────────────────┐ 
   100000 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠹⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡇⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠃⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⡼⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
          │⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀⠀⠀⠀⠀⠀⠀⢳⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
        0 │⢀⣀⣀⣀⣀⣀⣀⣀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠓⢄⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⠀⠀⠀⠀│ 
          └────────────────────────────────────────┘ 
          ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀80⠀ 
```

## Multiprocessing
Submodule `GraphTensorNetworks.SimpleMutiprocessing` provides a function [`GraphTensorNetworks.SimpleMultiprocessing.multiprocess_run`](@ref) function for simple multi-processing jobs.
Suppose we want to find the independence polynomial for multiple graphs with 4 processes.
We can create a file, e.g. named `run.jl` with the following content

```julia
using Distributed, GraphTensorNetworks.SimpleMultiprocessing
using Random, GraphTensorNetworks  # to avoid multi-precompiling
@everywhere using Random, GraphTensorNetworks

results = multiprocess_run(collect(1:10)) do seed
    Random.seed!(seed)
    n = 10
    @info "Graph size $n x $n, seed= $seed"
    g = random_diagonal_coupled_graph(n, n, 0.8)
    gp = Independence(g; optimizer=TreeSA(), simplifier=MergeGreedy())
    res = solve(gp, GraphPolynomial())[]
    return res
end

println(results)
```

One can run this script file with the following command
```bash
$ julia -p4 run.jl
      From worker 3:	[ Info: running argument 4 on device 3
      From worker 4:	[ Info: running argument 2 on device 4
      From worker 5:	[ Info: running argument 3 on device 5
      From worker 2:	[ Info: running argument 1 on device 2
      From worker 3:	[ Info: Graph size 10 x 10, seed= 4
      From worker 4:	[ Info: Graph size 10 x 10, seed= 2
      From worker 5:	[ Info: Graph size 10 x 10, seed= 3
      From worker 2:	[ Info: Graph size 10 x 10, seed= 1
      From worker 4:	[ Info: running argument 5 on device
      ...
```
You will see a vector of polynomials printed out.

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

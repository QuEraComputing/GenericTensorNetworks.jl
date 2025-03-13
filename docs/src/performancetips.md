# Performance Tips

## Overview
This guide provides strategies to optimize performance when using GenericTensorNetworks.jl. 
We'll cover:

1. Optimizing contraction orders
2. Using slicing techniques for large problems
3. Accelerating computations with specialized GEMM operations
4. Leveraging multiprocessing for parallel execution
5. Utilizing GPU acceleration
6. Performance benchmarks and comparisons

## 1. Optimizing Contraction Orders

Let's examine how to optimize contraction orders using the independent set problem on 3-regular graphs:

```julia
using GenericTensorNetworks, Graphs, Random
graph = random_regular_graph(120, 3)
iset = IndependentSet(graph)
problem = GenericTensorNetwork(iset; optimizer=TreeSA(
    sc_target=20, sc_weight=1.0, rw_weight=3.0, ntrials=10, βs=0.01:0.1:15.0, niters=20))
```

The `GenericTensorNetwork` constructor maps a problem to a tensor network with an optimized contraction order. The `optimizer` parameter specifies the algorithm to use:

### Available Optimizers:
- **TreeSA** (used above): A simulated annealing-based optimizer that often finds the smallest time/space complexity and supports slicing. See [arXiv: 2108.05665](https://arxiv.org/abs/2108.05665) for details.
- **GreedyMethod** (default): Fastest search but typically yields higher contraction complexity
- **KaHyParBipartite**: Uses hypergraph partitioning
- **SABipartite**: Simulated annealing on bipartite representation

The returned `problem` object contains a `code` field specifying the tensor network with optimized contraction order. For an independent set problem, the optimal contraction complexity is approximately 2^(tw(G)), where tw(G) is the [tree-width](https://en.wikipedia.org/wiki/Treewidth) of graph G.

### Analyzing Contraction Complexity

You can check the time, space, and read-write complexity with:

```julia
contraction_complexity(problem)
```

This returns log2 values of:
1. Number of multiplications
2. Number of elements in the largest tensor during contraction
3. Number of read-write operations to tensor elements

In our example, the computation requires approximately 2^21.9 multiplications, 2^20 read-write operations, and the largest tensor contains 2^17 elements.

### Memory Requirements

Different element types have different memory footprints:

```julia
sizeof(TropicalF64)
sizeof(TropicalF32)
sizeof(StaticBitVector{200,4})
sizeof(TruncatedPoly{5,Float64,Float64})
```

For a more accurate estimate of peak memory usage in bytes, use:

```julia
estimate_memory(problem, GraphPolynomial(; method=:finitefield))
estimate_memory(problem, GraphPolynomial(; method=:polynomial))
```

The finite field approach requires only 298 KB, while using the `Polynomial` type requires 71 MB.

> **Note**: 
> - Actual runtime memory can be several times larger than the maximum tensor size
> - For mutable element types like `ConfigEnumerator`, memory estimation functions may not accurately measure actual usage

## 2. Slicing Technique for Large Problems

For large-scale applications, you can slice over certain degrees of freedom to reduce space complexity. This approach loops and accumulates over selected degrees of freedom, resulting in smaller tensor networks inside the loop.

In the `TreeSA` optimizer, set `nslices` to a value greater than zero:

```julia
# Without slicing
problem = GenericTensorNetwork(iset; optimizer=TreeSA(βs=0.01:0.1:25.0, ntrials=10, niters=10))
contraction_complexity(problem)

# With slicing over 5 degrees of freedom
problem = GenericTensorNetwork(iset; optimizer=TreeSA(βs=0.01:0.1:25.0, ntrials=10, niters=10, nslices=5))
contraction_complexity(problem)
```

In this example, slicing over 5 degrees of freedom reduces space complexity by a factor of 32 (2^5), while increasing computation time by less than a factor of 2.

## 3. Accelerating Tropical Number Operations

You can significantly speed up Tropical number matrix multiplication when computing `SizeMax()` by using specialized GEMM routines from [TropicalGEMM](https://github.com/TensorBFS/TropicalGEMM.jl/):

```julia
using BenchmarkTools

# Without TropicalGEMM
@btime solve(problem, SizeMax())
# 91.630 ms (19203 allocations: 23.72 MiB)
# 0-dimensional Array{TropicalF64, 0}:
# 53.0ₜ

# With TropicalGEMM
using TropicalGEMM
@btime solve(problem, SizeMax())
# 8.960 ms (18532 allocations: 17.01 MiB)
# 0-dimensional Array{TropicalF64, 0}:
# 53.0ₜ
```

TropicalGEMM overrides the `LinearAlgebra.mul!` interface and takes effect immediately upon loading. This example shows more than 10x speedup on a single-threaded CPU. Performance can be further improved with [Julia multi-threading](https://docs.julialang.org/en/v1/manual/multi-threading/).

## 4. Multiprocessing for Parallel Execution

The `GenericTensorNetworks.SimpleMultiprocessing` submodule provides a convenient `multiprocess_run` function for parallel jobs. Here's how to find independence polynomials for multiple graphs using 4 processes:

```julia
# File: run.jl
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
```

Run this script with:
```bash
$ julia -p4 run.jl
```

## 5. GPU Acceleration

To run computations on a GPU, simply load CUDA and set the `usecuda` parameter:

```julia
using CUDA
# [ Info: OMEinsum loaded the CUDA module successfully

solve(problem, SizeMax(), usecuda=true)
# 0-dimensional CuArray{TropicalF64, 0, CUDA.Mem.DeviceBuffer}:
# 53.0ₜ
```

### GPU-Compatible Solution Properties:
- `SizeMax` and `SizeMin`
- `CountingAll`
- `CountingMax` and `CountingMin`
- `GraphPolynomial`
- `SingleConfigMax` and `SingleConfigMin`

## 6. Performance Benchmarks

We benchmarked performance on an Intel Xeon CPU E5-2686 v4 @ 2.30GHz (single thread) and a Tesla V100-SXM2 16GB GPU. The benchmark code is available in [our paper repository](https://github.com/GiggleLiu/NoteOnTropicalMIS/tree/master/benchmarks).

### Independent Set Problem Benchmarks
![benchmark-independent-set](assets/fig1.png)

These benchmarks show computation time for various independent set properties on random three-regular graphs:

**Figure (a)**: Time and space complexity vs. number of vertices
- Slicing was used for graphs with space complexity > 2^27 (above yellow line)

**Figure (b)**: Computation time for:
- MIS size calculation
- Counting all independent sets
- Counting MISs
- Counting sets of size α(G) and α(G)-1
- Finding 100 largest set sizes

**Figure (c)**: Computation time for independence polynomials:
- Fourier transformation method is fastest but may have round-off errors
- Finite field (GF(p)) approach has no round-off errors and works on GPU

**Figure (d)**: Configuration enumeration time:
- Bounding techniques improve performance by >10x for MIS enumeration
- Bounding also significantly reduces memory usage

### Maximal Independent Set Benchmarks
![benchmark-maximal-independent-set](assets/fig2.png)

**Figure (a)**: Time and space complexity for maximal IS problems
- Typically higher than for standard independent set problems

**Figure (b)**: Wall clock time comparison:
- Counting maximal ISs is much more efficient than enumeration
- Our tensor network approach is slightly faster than Bron-Kerbosch for enumerating maximal ISs



# Efficient Configuration Storage with Sum-Product Trees

## Overview
When dealing with combinatorial problems, the number of valid configurations can grow exponentially with problem size. The `SumProductTree` data structure provides a memory-efficient solution for storing and sampling from these large configuration spaces.

## The Sum-Product Tree Approach

A `SumProductTree` is a specialized data structure that:
- Uses polynomial memory to store an exponential number of configurations
- Represents configurations as a sum-product expression tree
- Enables lazy evaluation through depth-first search
- Supports efficient directed sampling from the configuration space

This approach is particularly valuable when working with large graphs where storing all configurations explicitly would be prohibitively expensive.

## Example: Independent Sets in Large Graphs

Let's examine how to use a `SumProductTree` for a large random regular graph:

```julia
using GenericTensorNetworks
graph = random_regular_graph(70, 3)
problem = GenericTensorNetwork(IndependentSet(graph); optimizer=TreeSA())
tree = solve(problem, ConfigsAll(; tree_storage=true))[]
```

For this 70-vertex graph, storing all independent sets explicitly would require approximately 256 TB of storage! However, the `SumProductTree` representation requires only a fraction of this memory.

## Sampling from the Configuration Space

One of the most powerful features of the `SumProductTree` is its ability to generate unbiased random samples from the configuration space:

```julia
samples = generate_samples(tree, 1000)
```

This generates 1000 random independent set configurations from our graph, allowing us to analyze statistical properties without enumerating the entire solution space.

## Statistical Analysis: Hamming Distance Distribution

With these samples, we can compute useful properties such as the Hamming distance distribution between configurations. The Hamming distance measures how many bit positions differ between two configurations.

```julia
using CairoMakie
dist = hamming_distribution(samples, samples)

# Create a bar plot of the distribution
fig = Figure()
ax = Axis(fig[1, 1]; xlabel="Hamming Distance", ylabel="Frequency")
barplot!(ax, 0:length(dist)-1, dist)
fig
```

This visualization reveals the structure of the solution space by showing how similar or dissimilar the independent set configurations tend to be to each other.

## Applications

The `SumProductTree` approach is particularly valuable for:
- Analyzing very large problem instances
- Estimating statistical properties of solution spaces
- Performing Monte Carlo sampling for approximation algorithms
- Studying the structure of configuration spaces without exhaustive enumeration

By combining compact representation with efficient sampling, `SumProductTree` enables analysis of problem instances that would otherwise be computationally intractable.

```@repl sumproduct
using GenericTensorNetworks
graph = random_regular_graph(70, 3)
problem = GenericTensorNetwork(IndependentSet(graph); optimizer=TreeSA());
tree = solve(problem, ConfigsAll(; tree_storage=true))[]
```
If one wants to store these configurations, he will need a hard disk of size 256 TB!
However, this sum-product binary tree structure supports efficient and unbiased direct sampling.

```@repl sumproduct
samples = generate_samples(tree, 1000)
```

With these samples, one can already compute useful properties like Hamming distance (see [`hamming_distribution`](@ref)) distribution. The following code visualizes this distribution with `CairoMakie`.

```@example sumproduct
using CairoMakie
dist = hamming_distribution(samples, samples)
# bar plot
fig = Figure()
ax = Axis(fig[1, 1]; xlabel="Hamming distance", ylabel="Frequency")
barplot!(ax, 0:length(dist)-1, dist)
fig
```
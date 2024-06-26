# Sum product representation for configurations
[`SumProductTree`](@ref) can use polynomial memory to store exponential number of configurations.
It is a sum-product expression tree to store [`ConfigEnumerator`](@ref) in a lazy style, where configurations can be extracted by depth first searching the tree with the `Base.collect` method.
Although it is space efficient, it is in general not easy to extract information from it due to the exponential large configuration space.
Directed sampling is one of its most important operations, with which one can get some statistic properties from it with an intermediate effort. For example, if we want to check some property of an intermediate scale graph, one can type
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
# Sum product representation for configurations
[`SumProductTree`](@ref) can use polynomial memory to store exponential number of configurations.
It is a sum-product expression tree to store [`ConfigEnumerator`](@ref) in a lazy style, where configurations can be extracted by depth first searching the tree with the `Base.collect` method.
Although it is space efficient, it is in general not easy to extract information from it due to the exponential large configuration space.
Directed sampling is one of its most important operations, with which one can get some statistic properties from it with an intermediate effort. For example, if we want to check some property of an intermediate scale graph, one can type
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

With these samples, one can already compute useful properties like Hamming distance (see [`hamming_distribution`](@ref)) distribution.

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

Here, the ``x``-axis is the Hamming distance and the ``y``-axis is the counting of pair-wise Hamming distances.
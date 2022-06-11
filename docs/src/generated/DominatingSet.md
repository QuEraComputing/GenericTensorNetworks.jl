```@meta
EditURL = "<unknown>/examples/DominatingSet.jl"
```

# Dominating set problem

!!! note
    It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

## Problem definition

In graph theory, a [dominating set](https://en.wikipedia.org/wiki/Dominating_set) for a graph ``G = (V, E)`` is a subset ``D`` of ``V`` such that every vertex not in ``D`` is adjacent to at least one member of ``D``.
The domination number ``\gamma(G)`` is the number of vertices in a smallest dominating set for ``G``.
The decision version of finding the minimum dominating set is an NP-complete.
In the following, we are going to solve the dominating set problem on the Petersen graph.

````@example DominatingSet
using GenericTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)
````

We can visualize this graph using the following function

````@example DominatingSet
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 1.0, i) for i=0:4]..., [rot15(0.0, 0.6, i) for i=0:4]...]

show_graph(graph; locs=locations)
````

## Generic tensor network representation
We can use [`DominatingSet`](@ref) to construct the tensor network for solving the dominating set problem as

````@example DominatingSet
problem = DominatingSet(graph; optimizer=TreeSA());
nothing #hide
````

### Theory (can skip)
Let ``G=(V,E)`` be the target graph that we want to solve.
The tensor network representation map a vertex ``v\in V`` to a boolean degree of freedom ``s_v\in\{0, 1\}``.
We defined the restriction on a vertex and its neighbouring vertices ``N(v)``:
```math
T(x_v)_{s_1,s_2,\ldots,s_{|N(v)|},s_v} = \begin{cases}
    0 & s_1=s_2=\ldots=s_{|N(v)|}=s_v=0,\\
    1 & s_v=0,\\
    x_v^{w_v} & \text{otherwise},
\end{cases}
```
where ``w_v`` is the weight of vertex ``v``.
This tensor means if both ``v`` and its neighbouring vertices are not in ``D``, i.e., ``s_1=s_2=\ldots=s_{|N(v)|}=s_v=0``,
this configuration is forbidden because ``v`` is not adjacent to any member in the set.
otherwise, if ``v`` is in ``D``, it has a contribution ``x_v^{w_v}`` to the final result.
One can check the contraction time space complexity of a [`DominatingSet`](@ref) instance by typing:

````@example DominatingSet
timespacereadwrite_complexity(problem)
````

## Solving properties

### Counting properties
##### Domination polynomial
The graph polynomial for the dominating set problem is known as the domination polynomial (see [arXiv:0905.2251](https://arxiv.org/abs/0905.2251)).
It is defined as
```math
D(G, x) = \sum_{k=0}^{\gamma(G)} d_k x^k,
```
where ``d_k`` is the number of dominating sets of size ``k`` in graph ``G=(V, E)``.

````@example DominatingSet
domination_polynomial = solve(problem, GraphPolynomial())[]
````

The domination number ``\gamma(G)`` can be computed with the [`SizeMin`](@ref) property:

````@example DominatingSet
domination_number = solve(problem, SizeMin())[]
````

Similarly, we have its counting [`CountingMin`](@ref):

````@example DominatingSet
counting_min_dominating_set = solve(problem, CountingMin())[]
````

### Configuration properties
##### finding minimum dominating set
One can enumerate all minimum dominating sets with the [`ConfigsMin`](@ref) property in the program.

````@example DominatingSet
min_configs = solve(problem, ConfigsMin())[].c

all(c->is_dominating_set(graph, c), min_configs)
````

````@example DominatingSet
show_gallery(graph, (2, 5); locs=locations, vertex_configs=min_configs)
````

Similarly, if one is only interested in computing one of the minimum dominating sets,
one can use the graph property [`SingleConfigMin`](@ref).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


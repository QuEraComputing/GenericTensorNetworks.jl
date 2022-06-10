```@meta
EditURL = "<unknown>/examples/SetCovering.jl"
```

# Set covering problem

!!! note
    It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

## Problem definition

The [set covering problem](https://en.wikipedia.org/wiki/Set_cover_problem) is a significant NP-hard problem in combinatorial optimization. Given a collection of elements, the set covering problem aims to find the minimum number of sets that incorporate (cover) all of these elements.
In the following, we will find the solution space properties for the camera location and stadium area example in the [Cornell University Computational Optimization Open Textbook](https://optimization.cbe.cornell.edu/index.php?title=Set_covering_problem).

````@example SetCovering
using GenericTensorNetworks, Graphs
````

The covering stadium areas of cameras are represented as the following sets.

````@example SetCovering
sets = [[1,3,4,6,7], [4,7,8,12], [2,5,9,11,13],
    [1,2,14,15], [3,6,10,12,14], [8,14,15],
    [1,2,6,11], [1,2,4,6,8,12]]
````

## Generic tensor network representation
We create a [`SetCovering`] instance that contains a generic tensor network as its `code` field.

````@example SetCovering
problem = SetCovering(sets);
nothing #hide
````

### Theory (can skip)
Let ``S`` be the target set covering problem that we want to solve.
For each set ``s \in S``, we associate it with a weight ``w_s`` to it.
The tensor network representation map a set ``s\in S`` to a boolean degree of freedom ``v_s\in\{0, 1\}``.
For each set ``s``, we defined a parameterized rank-one tensor indexed by ``v_s`` as
```math
W(x_s^{w_s}) = \left(\begin{matrix}
    1 \\
    x_s^{w_s}
    \end{matrix}\right)
```
where ``x_s`` is a variable associated with ``s``.
For each unique element ``a``, we defined the constraint over all sets containing it ``N(a) = \{s | s \in S \land a\in s\}``:
```math
B_{s_1,s_2,\ldots,s_{|N(a)|}} = \begin{cases}
    0 & s_1=s_2=\ldots=s_{|N(a)|}=0,\\
    1 & \text{otherwise}.
\end{cases}
```
This tensor means if none of the sets containing element ``a`` are included, then this configuration is forbidden,
One can check the contraction time space complexity of a [`SetCovering`](@ref) instance by typing:

````@example SetCovering
timespacereadwrite_complexity(problem)
````

## Solving properties

### Counting properties
##### The "graph" polynomial
The graph polynomial for the set covering problem is defined as
```math
P(S, x) = \sum_{k=0}^{|S|} c_k x^k,
```
where ``c_k`` is the number of configurations having ``k`` sets.

````@example SetCovering
covering_polynomial = solve(problem, GraphPolynomial())[]
````

The minimum number of sets that covering the set of elements can be computed with the [`SizeMin`](@ref) property:

````@example SetCovering
min_cover_size = solve(problem, SizeMin())[]
````

Similarly, we have its counting [`CountingMin`](@ref):

````@example SetCovering
counting_minimum_setcovering = solve(problem, CountingMin())[]
````

### Configuration properties
##### Finding minimum set covering
One can enumerate all minimum set covering with the [`ConfigsMin`](@ref) property in the program.

````@example SetCovering
min_configs = solve(problem, ConfigsMin())[].c
````

Hence the two optimal solutions are ``\{z_1, z_3, z_5, z_6\}`` and ``\{z_2, z_3, z_4, z_5\}``.
The correctness of this result can be checked with the [`is_set_covering`](@ref) function.

````@example SetCovering
all(c->is_set_covering(sets, c), min_configs)
````

Similarly, if one is only interested in computing one of the minimum set coverings,
one can use the graph property [`SingleConfigMin`](@ref).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


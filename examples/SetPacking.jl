# # Set packing problem

# !!! note
#     It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

# ## Problem definition

# The [set packing problem](https://en.wikipedia.org/wiki/Set_packing) is generalization of the [`IndependentSet`](@ref) problem from the simple graph to the multigraph.
# Suppose one has a finite set ``S`` and a list of subsets of ``S``. Then, the set packing problem asks if some ``k`` subsets in the list are pairwise disjoint.
# In the following, we will find the solution space properties for the set in the [Set covering problem](@ref).

using GraphTensorNetworks, Graphs

# The packing stadium areas of cameras are represented as the following sets.

sets = [[1,3,4,6,7], [4,7,8,12], [2,5,9,11,13],
    [1,2,14,15], [3,6,10,12,14], [8,14,15], 
    [1,2,6,11], [1,2,4,6,8,12]]

# ## Generic tensor network representation
# We create a [`SetPacking`] instance that contains a generic tensor network as its `code` field.
problem = SetPacking(sets);

# ### Theory (can skip)
# Let ``S`` be the target set packing problem that we want to solve.
# For each set ``s \in S``, we associate it with a weight ``w_s`` to it.
# The tensor network representation map a set ``s\in S`` to a boolean degree of freedom ``v_s\in\{0, 1\}``.
# For each set ``s``, we defined a parameterized rank-one tensor indexed by ``v_s`` as
# ```math
# W(x_s^{w_s}) = \left(\begin{matrix}
#     1 \\
#     x_s^{w_s}
#     \end{matrix}\right)
# ```
# where ``x_s`` is a variable associated with ``s``.
# For each unique element ``a``, we defined the constraint over all sets containing it ``N(a) = \{s | s \in S \land a\in s\}``:
# ```math
# B_{s_1,s_2,\ldots,s_{|N(a)|}} = \begin{cases}
#     0 & s_1+s_2+\ldots+s_{|N(a)|} > 1,\\
#     1 & \text{otherwise}.
# \end{cases}
# ```
# This tensor means if in a configuration, two sets contain the element ``a``, then this configuration is forbidden,
# One can check the contraction time space complexity of a [`SetPacking`](@ref) instance by typing:

timespacereadwrite_complexity(problem)

# ## Solving properties

# ### Counting properties
# ##### The "graph" polynomial
# The graph polynomial for the set packing problem is defined as
# ```math
# P(S, x) = \sum_{k=0}^{\alpha(S)} c_k x^k,
# ```
# where ``c_k`` is the number of configurations having ``k`` sets, and ``\alpha(S)`` is the maximum size of the packing.

packing_polynomial = solve(problem, GraphPolynomial())[]

# The maximum number of sets that packing the set of elements can be computed with the [`SizeMax`](@ref) property:
max_packing_size = solve(problem, SizeMax())[]

# Similarly, we have its counting [`CountingMax`](@ref):
counting_maximum_set_packing = solve(problem, CountingMax())[]

# ### Configuration properties
# ##### Finding maximum set packing
# One can enumerate all maximum set packing with the [`ConfigsMax`](@ref) property in the program.
max_configs = solve(problem, ConfigsMax())[].c

# Hence the only optimal solution is ``\{z_1, z_3, z_6\}`` that has size 3.
# The correctness of this result can be checked with the [`is_set_packing`](@ref) function.

all(c->is_set_packing(sets, c), max_configs)

# Similarly, if one is only interested in computing one of the maximum set packing,
# one can use the graph property [`SingleConfigMax`](@ref).

# # Maximal independent set problem

# !!! note
#     It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

# ## Problem definition

# In graph theory, a [maximal independent set](https://en.wikipedia.org/wiki/Maximal_independent_set) is an independent set that is not a subset of any other independent set.
# It is different from maximum independent set because it does not require the set to have the max size.
# In the following, we are going to solve the maximal independent set problem on the Petersen graph.

using GenericTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a
locations = [[rot15(0.0, 60.0, i) for i=0:4]..., [rot15(0.0, 30, i) for i=0:4]...]
show_graph(graph, locations; format=:svg)

# ## Generic tensor network representation
# We can use [`MaximalIS`](@ref) to construct the tensor network for solving the maximal independent set problem as
maximalis = MaximalIS(graph)

# The tensor network representation of the maximal independent set problem can be obtained by
problem = GenericTensorNetwork(maximalis)

# ### Theory (can skip)
# Let ``G=(V,E)`` be the target graph that we want to solve.
# The tensor network representation map a vertex ``v\in V`` to a boolean degree of freedom ``s_v\in\{0, 1\}``.
# We defined the restriction on its neighborhood ``N(v)``:
# ```math
# T(x_v)_{s_1,s_2,\ldots,s_{|N(v)|},s_v} = \begin{cases}
#     s_vx_v^{w_v} & s_1=s_2=\ldots=s_{|N(v)|}=0,\\
#     1-s_v& \text{otherwise}.
# \end{cases}
# ```
# The first case corresponds to all the neighborhood vertices of ``v`` are not in ``I_{m}``, then ``v`` must be in ``I_{m}`` and contribute a factor ``x_{v}^{w_v}``,
# where ``w_v`` is the weight of vertex ``v``.
# Otherwise, if any of the neighboring vertices of ``v`` is in ``I_{m}``, ``v`` must not be in ``I_{m}`` by the independence requirement.

# Its contraction time space complexity of a [`MaximalIS`](@ref) instance is no longer determined by the tree-width of the original graph ``G``.
# It is often harder to contract this tensor network than to contract the one for regular independent set problem.

contraction_complexity(problem)

# ## Solving properties

# ### Counting properties
# ##### maximal independence polynomial
# The graph polynomial defined for the maximal independent set problem is
# ```math
# I_{\rm max}(G, x) = \sum_{k=0}^{\alpha(G)} b_k x^k,
# ```
# where ``b_k`` is the number of maximal independent sets of size ``k`` in graph ``G=(V, E)``.

maximal_indenpendence_polynomial = solve(problem, GraphPolynomial())[]

# One can see the first several coefficients are 0, because it only counts the maximal independent sets, 
# The minimum maximal independent set size is also known as the independent domination number.
# It can be computed with the [`SizeMin`](@ref) property:
independent_domination_number = solve(problem, SizeMin())[]

# Similarly, we have its counting [`CountingMin`](@ref):
counting_min_maximal_independent_set = solve(problem, CountingMin())[]

# ### Configuration properties
# ##### finding all maximal independent set
maximal_configs = solve(problem, ConfigsAll())[]

all(c->is_maximal_independent_set(graph, c), maximal_configs)

#

show_gallery(graph, (3, 5); locs=locations, vertex_configs=maximal_configs, format=:svg)

# This result should be consistent with that given by the [Bron Kerbosch algorithm](https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm) on the complement of Petersen graph.
cliques = maximal_cliques(complement(graph))

# For sparse graphs, the generic tensor network approach is usually much faster and memory efficient than the Bron Kerbosch algorithm.

# ##### finding minimum maximal independent set
# It is the [`ConfigsMin`](@ref) property in the program.
minimum_maximal_configs = solve(problem, ConfigsMin())[].c

show_gallery(graph, (2, 5); locs=locations, vertex_configs=minimum_maximal_configs, format=:svg)

# Similarly, if one is only interested in computing one of the minimum sets,
# one can use the graph property [`SingleConfigMin`](@ref).

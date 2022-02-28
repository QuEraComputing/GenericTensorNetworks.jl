# # Dominating set problem

# !!! note
#     It is recommended to read the [Independent set problem](@ref) tutorial first to know more about
#     * how to optimize the tensor network contraction order,
#     * what graph properties are available and how to select correct method to compute graph properties,
#     * how to compute weighted graphs and handle open vertices.

# ## Problem definition

# In graph theory, a [dominating set](https://en.wikipedia.org/wiki/Dominating_set) for a graph ``G = (V, E)`` is a subset ``D`` of ``V`` such that every vertex not in ``D`` is adjacent to at least one member of ``D``.
# The domination number ``\gamma(G)`` is the number of vertices in a smallest dominating set for ``G``.
# The decision version of finding the minimum dominating set is an NP-complete.
# In the following, we are going to solve the dominating set problem on the Petersen graph.

using GraphTensorNetworks, Graphs, Compose

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 1.0, i) for i=0:4]..., [rot15(0.0, 0.6, i) for i=0:4]...]

show_graph(graph; locs=locations)

# ## Tensor network representation
# Let ``G=(V,E)`` be the target graph that we want to solve.
# The tensor network representation map a vertex ``v\in V`` to a boolean degree of freedom ``s_v\in\{0, 1\}``.
# We defined the restriction on a vertex and its neighbouring vertices ``N(v)``:
# ```math
# T(x_v)_{s_1,s_2,\ldots,s_{|N(v)|},s_v} = \begin{cases}
#     0 & s_1=s_2=\ldots=s_{|N(v)|}=s_v=0,\\
#     1 & s_v=0,\\
#     x_v^{w_v} & \text{otherwise},
# \end{cases}
# ```
# where ``w_v`` is the weight of vertex ``v``.
# This tensor means if both ``v`` and its neighbouring vertices are not in ``D``, i.e., ``s_1=s_2=\ldots=s_{|N(v)|}=s_v=0``,
# this configuration is forbidden because ``v`` is not adjacent to any member in the set.
# otherwise, if ``v`` is in ``D``, it has a contribution ``x_v^{w_v}`` to the final result.
# We can use [`DominatingSet`](@ref) to construct the tensor network for solving the dominating set problem as
problem = DominatingSet(graph; optimizer=TreeSA());

# One can check the contraction time space complexity of a [`DominatingSet`](@ref) instance by typing:

timespacereadwrite_complexity(problem)

# Results are `log2` values of time (number of iterations),
# space (number of items in the largest tensor)
# and read-write (number of read-write of operations to elements).

# ## Solving properties

# ### Counting properties
# ##### Domination polynomial
# The graph polynomial for the dominating set problem is known as the domination polynomial (see [arXiv:0905.2251](https://arxiv.org/abs/0905.2251)).
# It is defined as
# ```math
# D(G, x) = \sum_{k=0}^{\gamma(G)} d_k x^k,
# ```
# where ``d_k`` is the number of dominating sets of size ``k`` in graph ``G=(V, E)``.

domination_polynomial = solve(problem, GraphPolynomial())[]

# The domination number ``\gamma(G)`` can be computed with the [`SizeMin`](@ref) property:
domination_number = solve(problem, SizeMin())[]

# Similarly, we have its counting [`CountingMin`](@ref):
counting_min_dominating_set = solve(problem, CountingMin())[]

# ### Configuration properties
# ##### finding all dominating set
# One can enumerate all minimum dominating sets with the [`ConfigsMin`](@ref) property in the program.
min_configs = solve(problem, ConfigsMin())[].c

all(c->is_dominating_set(graph, c), min_configs)

# ##### finding minimum dominating set

imgs = ntuple(k->show_graph(graph;
                locs=locations, scale=0.25,
                vertex_colors=[iszero(min_configs[k][i]) ? "white" : "red"
                for i=1:nv(graph)]), length(min_configs));

Compose.set_default_graphic_size(18cm, 8cm); Compose.compose(context(),
     ntuple(k->(context((mod1(k,5)-1)/5, ((k-1)÷5)/2, 1.2/5, 1.0/2), imgs[k]), 10)...)

# Similarly, if one is only interested in computing one of the minimum dominating sets,
# one can use the graph property [`SingleConfigMin`](@ref).
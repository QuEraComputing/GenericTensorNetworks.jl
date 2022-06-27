# # Spin-glass problem (Cutting problem)

# !!! note
#     It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

# ## Problem definition
# Let ``G=(V, E)`` be a graph, the [spin-glass](https://en.wikipedia.org/wiki/Spin_glass) problem in physics is characterized by the following energy function
# ```math
# H = - \sum_{ij \in E} J_{ij} s_i s_j + \sum_{i \in V} h_i s_i,
# ```
# where ``h_i`` is an onsite energy term associated with spin ``s_i \in \{0, 1\}``, and ``J_{ij}`` is the coupling strength between spins ``s_i`` and ``s_j``.
#
# The spin glass problem very close related to the cutting problem in graph theory.
# A [cut](https://en.wikipedia.org/wiki/Cut_(graph_theory)) is a partition of the vertices of a graph into two disjoint subsets.
# Finding the maximum cut (the spin glass maximum energy) is NP-Hard, where a maximum cut is a cut whose size is at least the size of any other cut,
# where the size of a cut is the number of edges (or the sum of weights on edges) crossing the cut.

using GenericTensorNetworks, Graphs

# In the following, we are going to defined an spin glass problem for the Petersen graph.

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 2.0, i) for i=0:4]..., [rot15(0.0, 1.0, i) for i=0:4]...]

show_graph(graph; locs=locations, format=:svg)

# ## Generic tensor network representation
# We define the spin glass problem as
problem = SpinGlass(graph);

# ### Theory (can skip)
#
# For a vertex ``v\in V``, we define a boolean degree of freedom ``s_v\in\{0, 1\}``.
# Then the spin glass problem can be encoded to tensor networks by mapping an edge ``(i,j)\in E`` to an edge matrix labelled by ``s_is_j``
# ```math
# B(x_{\langle i, j\rangle}) = \left(\begin{matrix}
#     1 & x_{\langle i, j\rangle}^{w_{\langle i,j \rangle}}\\
#     x_{\langle i, j\rangle}^{w_{\langle i,j \rangle}} & 1
# \end{matrix}\right),
# ```
# If and only if the spin configuration is anti-parallel on edge ``(i, j)``,
# this tensor contributes a factor ``x_{\langle i, j\rangle}^{w_{\langle i,j \rangle}}``,
# where ``w_{\langle i,j\rangle}`` is the weight of this edge.
# Similar to other problems, we can define a polynomial about edges variables by setting ``x_{\langle i, j\rangle} = x``,
# where its k-th coefficient is two times the number of configurations with energy (cut size) k.

# Its contraction time space complexity is ``2^{{\rm tw}(G)}``, where ``{\rm tw(G)}`` is the [tree-width](https://en.wikipedia.org/wiki/Treewidth) of ``G``.

# ## Solving properties
# ### Maximum energy ``E^*(G)``
Emax = solve(problem, SizeMax())[]

# ### Counting properties
# ##### graph polynomial
# The graph polynomial defined for the spin glass problem is
# ```math
# C(G, x) = \sum_{k=0}^{E^*(G)} c_k x^k,
# ```
# where ``\alpha(G)`` is the maximum independent set size, 
# ``c_k/2`` is the number of anti-parallel edges (cuts) of size ``k`` in graph ``G=(V,E)``.
# Since the variable ``x`` is defined on edges,
# the coefficients of the polynomial is the number of configurations having different number of anti-parallel edges.
max_config = solve(problem, GraphPolynomial())[]

# ### Configuration properties
# ##### finding one solution with highest energy
max_vertex_config = solve(problem, SingleConfigMax())[].c.data

Emax_verify = spinglass_energy(graph, max_vertex_config)

# You should see a consistent result as above `Emax`.

show_graph(graph; locs=locations, vertex_colors=[
        iszero(max_vertex_config[i]) ? "white" : "red" for i=1:nv(graph)], format=:svg)

# where a red vertice and a white vertice correspond to a spin having value 1 and 0 respectively.

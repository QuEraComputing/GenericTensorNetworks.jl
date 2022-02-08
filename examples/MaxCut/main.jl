# # Cutting problem (Spin-glass problem)

# !!! note
#     This tutorial only covers the cutting problem specific features,
#     It is recommended to read the [Independent set problem](@ref) tutorial too to know more about
#     * how to optimize the tensor network contraction order,
#     * what are the other graph properties computable,
#     * how to select correct method to compute graph properties,
#     * how to compute weighted graphs and handle open vertices.

# ## Introduction
using GraphTensorNetworks, Graphs

# Please check the docstring of [`MaxCut`](@ref) for the definition of the vertex cutting problem.
@doc MaxCut

# In the following, we are going to defined an cutting problem for the Petersen graph.

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 1.0, i) for i=0:4]..., [rot15(0.0, 0.6, i) for i=0:4]...]

show_graph(graph; locs=locations)

# Then we define the cutting problem as
problem = MaxCut(graph);

# ## Solving properties

# ### Maximum cut size ``\gamma(G)``
max_cut_size = solve(problem, SizeMax())[]

# ### Counting properties
# ##### graph polynomial
# Since the variable ``x`` is defined on edges,
# hence the coefficients of the polynomial is the number of configurations having different number of anti-parallel edges.
max_config = solve(problem, GraphPolynomial())[]

# ### Configuration properties
# ##### finding one max cut solution
max_edge_config = solve(problem, SingleConfigMax())[]

# These configurations are defined on edges, we need to find a valid assignment on vertices
max_vertex_config = cut_assign(graph, max_edge_config.c.data)

max_cut_size_verify = cut_size(graph, max_vertex_config)

# You should see a consistent result as above `max_cut_size`.

show_graph(graph; locs=locations, colors=[
        iszero(max_vertex_config[i]) ? "white" : "red" for i=1:nv(graph)])
# # Coloring problem

# !!! note
#     This tutorial only covers the coloring problem specific features,
#     It is recommended to read the [Independent set problem](@ref) tutorial too to know more about
#     * how to optimize the tensor network contraction order,
#     * what are the other graph properties computable,
#     * how to select correct method to compute graph properties,
#     * how to compute weighted graphs and handle open vertices.

# ## Introduction
using GraphTensorNetworks, Graphs

# Please check the docstring of [`Coloring`](@ref) for the definition of the coloring problem.
@doc Coloring

# In the following, we are going to defined a 3-coloring problem for the Petersen graph.

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 1.0, i) for i=0:4]..., [rot15(0.0, 0.6, i) for i=0:4]...]

show_graph(graph; locs=locations)

# Then we define the cutting problem as
problem = Coloring{3}(graph);

# ## Solving properties
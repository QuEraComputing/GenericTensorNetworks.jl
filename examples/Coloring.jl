# # Graph Coloring Problem
#
# ## Overview
# Graph coloring is a fundamental problem in graph theory where we assign colors to vertices
# such that no adjacent vertices share the same color. This example demonstrates solving
# a 3-coloring problem on the Petersen graph using tensor networks.

using GenericTensorNetworks, Graphs

# Create a Petersen graph instance
graph = Graphs.smallgraph(:petersen)

# Define vertex layout for visualization
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a
locations = [[rot15(0.0, 60.0, i) for i=0:4]..., [rot15(0.0, 30, i) for i=0:4]...]
show_graph(graph, locations; format=:svg)

# ## Tensor Network Formulation
# We represent the 3-coloring problem using tensor networks:
coloring = Coloring{3}(graph)

constraints(coloring)

#

objectives(coloring)

# Convert the coloring problem to a tensor network:
problem = GenericTensorNetwork(coloring)

# ## Mathematical Background
# Type [`Coloring`](@ref) can be used for constructing the tensor network with optimized contraction order for a coloring problem.
# Let us use 3-coloring problem defined on vertices as an example.
# For a vertex ``v``, we define the degrees of freedom ``c_v\in\{1,2,3\}`` and a vertex tensor labelled by it as
# ```math
# W(v) = \left(\begin{matrix}
#     1\\
#     1\\
#     1
# \end{matrix}\right).
# ```
# For an edge ``(u, v)``, we define an edge tensor as a matrix labelled by ``(c_u, c_v)`` to specify the constraint
# ```math
# B = \left(\begin{matrix}
#     1 & x & x\\
#     x & 1 & x\\
#     x & x & 1
# \end{matrix}\right).
# ```
# The number of possible colorings can be obtained by contracting this tensor network by setting vertex tensor elements to 1.
#
# We can check the time, space and read-write complexities by typing:

contraction_complexity(problem)

# For more information about how to improve the contraction order, please check the [Performance Tips](@ref).

# ## Solution Analysis
# ### 1. Count All Valid Colorings
num_of_coloring = solve(problem, CountingMax())[]

# ### 2. Find One Valid Coloring
single_solution = solve(problem, SingleConfigMax())[]
coloring_config = read_config(single_solution)

# Verify the solution is valid
is_vertex_coloring(graph, coloring_config)

# Visualize the coloring solution
vertex_color_map = Dict(0=>"red", 1=>"green", 2=>"blue")
show_graph(graph, locations; format=:svg, vertex_colors=[vertex_color_map[Int(c)]
     for c in coloring_config])

# ## Edge Coloring Analysis
# Let's examine the same problem on the line graph (where edges become vertices)
linegraph = line_graph(graph)

# Visualize the line graph
show_graph(linegraph, [(locations[e.src] .+ locations[e.dst])
     for e in edges(graph)]; format=:svg)

# Attempt to solve 3-coloring on the line graph
lineproblem = Coloring{3}(linegraph)
num_of_coloring = solve(GenericTensorNetwork(lineproblem), CountingMax())[]
read_size_count(num_of_coloring)

# Note: The maximum size of 28 is less than the number of edges in the line graph,
# proving that no valid 3-coloring exists for the edges of a Petersen graph.

# ## More APIs
# The [Independent Set Problem](@ref) chapter has more examples on how to use the APIs.
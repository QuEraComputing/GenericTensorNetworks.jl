# # Solving independent set for a weighted unit disk graph
using GraphTensorNetworks, Graphs

# This example shows how to check the MIS degeneracy (another way of saying counting) of a weight unit disk graph.
# Let us construct a unit disk graph by specifying the locations of nodes, the unit disk radius is 1.5,
# which means two vertices within distance 1.5 are connected.
# For each vertex, we assign a weight to it.
locations = [(6,-3),(1, -1), (0,0), (6,-2), (1,1), (2,0),
     (2,2), (2,-2), (3,1), (3,-2), (4,1), (4,-1),
     (5, 1), (5, -1), (6,0), (6,-1), (7,-1), (7, -4),
     (8, -2), (8, -4), (9,-3)]


weights = [0.7439015222121296, 0.722970984162338,
     0.9502792990276312, 0.6617352332568173, 
     0.8592866066961992, 1.0, 1.0, 1.0, 1.0, 
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
     1.0, 1.0, 1.0, 1.0, 1.0]

graph = unit_disk_graph(locations, 1.5)

show_graph(graph; locs=locations)

# ## Find the best configurations
# We can easily check the MIS degeneracy by computing the [`CountingMax`](@ref) property.
counting_mapped = solve(IndependentSet(graph; weights=weights), CountingMax())[]

# The MIS degeneracy is the second field, which is 3.
# We can compute the [`ConfigsMax`](@ref) to enumerate all configurations with maximum independent set size.

configs_mapped = solve(IndependentSet(graph; weights=weights), ConfigsMax())[]

# * Solution 1
show_graph(graph; locs=locations, vertex_colors=
          [iszero(configs_mapped.c[1][i]) ? "white" : "red" for i=1:nv(graph)])

# * Solution 2
show_graph(graph; locs=locations, vertex_colors=
          [iszero(configs_mapped.c[2][i]) ? "white" : "red" for i=1:nv(graph)])

# * Solution 3
show_graph(graph; locs=locations, vertex_colors=
          [iszero(configs_mapped.c[3][i]) ? "white" : "red" for i=1:nv(graph)])
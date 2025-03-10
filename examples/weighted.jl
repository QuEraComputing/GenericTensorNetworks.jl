# # Weighted Problems
#
# ## Overview
# Many optimization problems involve weights associated with vertices or edges.
# This example demonstrates how to:
# 
# * Define weighted problem instances
# * Solve for optimal configurations
# * Analyze the energy spectrum of solutions
# * Visualize weighted solutions
#
# We'll use the Maximum Weighted Independent Set problem on the Petersen graph as our example.

using GenericTensorNetworks, GenericTensorNetworks.ProblemReductions, Graphs

# Create a Petersen graph instance
graph = Graphs.smallgraph(:petersen)

# ## Defining Weighted Problems
# Create a weighted independent set problem where each vertex i has weight i
problem = GenericTensorNetwork(IndependentSet(graph, collect(1:10)))

# Examine the weights assigned to each vertex
GenericTensorNetworks.weights(problem)

# The tensor labels associated with these weights can be accessed via:
ProblemReductions.local_solution_spec(problem.problem)

# Note: You can use a vector for custom weights or `UnitWeight()` for unweighted problems.
# Most solution space properties that work for unweighted graphs also work for weighted graphs.

# ## Finding Optimal Solutions
# Find the maximum weighted independent set:
max_config_weighted = solve(problem, SingleConfigMax())[]

# ## Visualizing Solutions
# Define vertex layout for visualization
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a
locations = [[rot15(0.0, 60.0, i) for i=0:4]..., [rot15(0.0, 30, i) for i=0:4]...]

# Visualize the maximum weighted independent set
show_graph(graph, locations; format=:svg, vertex_colors=
    [iszero(max_config_weighted.c.data[i]) ? "white" : "red" for i=1:nv(graph)])

# Note: The `GraphPolynomial` property is the only solution space property that cannot
# be defined for general real-weighted graphs (though it works for integer-weighted graphs).

# ## Analyzing the Energy Spectrum
# For weighted problems, it's often useful to examine the "energy spectrum" -
# the distribution of weights across different configurations.

# Compute the 10 largest weights and their configurations:
spectrum = solve(problem, SizeMax(10))[]

# The result is an `ExtendedTropical` object containing the ordered weights:
spectrum.orders

# Each element in `orders` is a `Tropical` number representing a solution weight.

# ## Finding Multiple Top Solutions
# Find the 5 independent sets with the highest weights:
max5_configs = read_config(solve(problem, SingleConfigMax(5))[])

# The return value contains `CountingTropical{Float64,ConfigSampler}` objects
# that store both the weights and configurations.

# Visualize these top 5 configurations:
show_configs(graph, locations, [max5_configs[j] for i=1:1, j=1:5]; padding_left=20)
# Each configuration represents an independent set with one of the 5 highest total weights.
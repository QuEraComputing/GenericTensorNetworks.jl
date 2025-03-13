# # Open and Fixed Degrees of Freedom
#
# ## Overview
# When analyzing complex systems, we often need to examine specific variables while
# marginalizing or conditioning on others. This example demonstrates two approaches:
#
# 1. **Open degrees of freedom**: Compute marginals over selected variables
# 2. **Fixed degrees of freedom**: Compute conditional values by fixing certain variables
#
# We'll illustrate these concepts using the Maximum Independent Set (MIS) problem on the Petersen graph.

using GenericTensorNetworks, Graphs

# Create a Petersen graph instance
graph = Graphs.smallgraph(:petersen)

# ## Approach 1: Open Degrees of Freedom
# ### Computing Marginals
# The `openvertices` parameter allows us to compute marginals over specified vertices.
# Here we compute the MIS tropical tensor with open vertices 1, 2, and 3:

problem = GenericTensorNetwork(IndependentSet(graph); openvertices=[1,2,3])
marginal = solve(problem, SizeMax())

# The result is a rank-3 tensor where each element represents the maximum independent set size
# for a specific configuration of the open vertices. This tensor is known as the MIS tropical tensor,
# which has applications in tropical tensor analysis.
#
# Each index corresponds to a vertex state (0 or 1), and the tensor value gives the
# maximum achievable independent set size given those fixed vertex states.

# ## Approach 2: Fixed Degrees of Freedom
# ### Computing Conditional Values
# The `fixedvertices` parameter allows us to condition on specific vertex assignments.
# We can achieve the same result as above by systematically fixing vertices to different values:

problem = GenericTensorNetwork(IndependentSet(graph); fixedvertices=Dict(1=>0, 2=>0, 3=>0))

# Create a tensor to store results for all possible configurations of vertices 1, 2, and 3
output = zeros(TropicalF64, 2, 2, 2)

# Compute MIS size for each possible configuration of the three vertices
marginal_alternative = map(CartesianIndices((2,2,2))) do ci
    problem.fixedvertices[1] = ci.I[1]-1  # Convert from 1-indexed to 0-indexed
    problem.fixedvertices[2] = ci.I[2]-1
    problem.fixedvertices[3] = ci.I[3]-1
    output[ci] = solve(problem, SizeMax())[]
end

# Both approaches produce the same marginal information for vertices 1, 2, and 3.

# ## Performance Considerations
# While both approaches yield the same results, their computational efficiency can differ significantly.
# The `openvertices` approach allows the contraction order optimizer to consider these degrees of freedom
# during optimization, potentially leading to more efficient contraction paths.
#
# Choose the appropriate method based on your specific needs:
# * Use `openvertices` when you need marginals over multiple configurations
# * Use `fixedvertices` when you need to condition on specific configurations or when
#   exploring a small number of fixed assignments
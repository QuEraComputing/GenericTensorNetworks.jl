# # Open and fixed degrees of freedom
# Open degrees of freedom is useful when one want to get the marginal about certain degrees of freedom.
# When one specifies the `openvertices` keyword argument in [`solve`](@ref) function as a tuple of vertices, the output will be a tensor that can be indexed by these degrees of freedom.
# Let us use the maximum independent set problem on Petersen graph as an example.
#
using GenericTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)

# The following code computes the MIS tropical tensor (reference to be added) with open vertices 1, 2 and 3.
problem = GenericTensorNetwork(IndependentSet(graph); openvertices=[1,2,3]);

marginal = solve(problem, SizeMax())

# The return value is a rank-3 tensor, with its elements being the MIS sizes under different configuration of open vertices.
# For the maximum independent set problem, this tensor is also called the MIS tropical tensor, which can be useful in the MIS tropical tensor analysis (reference to be added).

# One can also specify the fixed degrees of freedom by providing the `fixedvertices` keyword argument as a `Dict`, which can be used to get conditioned probability.
# For example, we can use the following code to do the same calculation as using `openvertices`.
problem = GenericTensorNetwork(IndependentSet(graph); fixedvertices=Dict(1=>0, 2=>0, 3=>0));

output = zeros(TropicalF64,2,2,2);

marginal_alternative = map(CartesianIndices((2,2,2))) do ci
    problem.fixedvertices[1] = ci.I[1]-1
    problem.fixedvertices[2] = ci.I[2]-1
    problem.fixedvertices[3] = ci.I[3]-1
    output[ci] = solve(problem, SizeMax())[]
end

# One can easily check this one also gets the correct marginal on vertices 1, 2 and 3.
# As a reminder, their computational hardness can be different, because the contraction order optimization program can optimize over open degrees of freedom.
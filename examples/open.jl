# # Open degrees of freedom
# Open degrees of freedom is useful when the graph under consideration is an induced subgraph of another graph.
# The vertices connected to the rest part of the parent graph can not be summed over directly, which can be specified with the `openvertices` keyword argument in the graph problem constructor.
# Let us use the maximum independent set problem on Petersen graph as an example.
#
using GenericTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)

# The following code computes the MIS tropical tensor (reference to be added) with open vertices 1, 2 and 3.
problem = IndependentSet(graph; openvertices=[1,2,3])

mis_tropical_tensor = solve(problem, SizeMax())

# The return value is a rank-3 tensor, with its elements being the MIS sizes under different configuration of open vertices.
# For the maximum independent set problem, this tensor is also called the MIS tropical tensor, which can be useful in the MIS tropical tensor analysis (reference to be added).

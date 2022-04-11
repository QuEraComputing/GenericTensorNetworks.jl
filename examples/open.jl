# # Open degree of freedoms
# Let us use the maximum independent set problem on Petersen graph as an example.
#
using GenericTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)

# The following code computes the MIS tropical tensor (reference to be added) with open vertices 1, 2 and 3.
problem = IndependentSet(graph; openvertices=[1,2,3])

mis_tropical_tensor = solve(problem, SizeMax())

# The MIS tropical tensor shows the MIS size under different configuration of open vertices.
# It is useful in MIS tropical tensor analysis.
# We can compatify (reference to be added) this MIS-Tropical tensor by typing

mis_compactify!(mis_tropical_tensor)

# It will eliminate some entries having no contribution to the MIS size when embeding this local graph into a larger one.

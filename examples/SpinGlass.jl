# # Spin-glass problem

# !!! note
#     It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

# ## Problem definition
# Let ``G=(V, E)`` be a graph, the [spin-glass](https://en.wikipedia.org/wiki/Spin_glass) problem in physics is characterized by the following energy function
# ```math
# H = \sum_{ij \in E} J_{ij} s_i s_j + \sum_{i \in V} h_i s_i,
# ```
# where ``h_i`` is an onsite energy term associated with spin ``s_i \in \{-1, 1\}``, and ``J_{ij}`` is the coupling strength between spins ``s_i`` and ``s_j``.
# In the program, we use boolean variable `n_i = \frac{1-s_i}{2}` to represent a spin configuration.

using GenericTensorNetworks, Graphs

# In the following, we are going to defined an spin glass problem for the Petersen graph.

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 2.0, i) for i=0:4]..., [rot15(0.0, 1.0, i) for i=0:4]...]

show_graph(graph; locs=locations, format=:svg)

# ## Generic tensor network representation
# We define an anti-ferromagnetic spin glass problem as
problem = SpinGlass(graph; J=fill(-1, ne(g)));

# ### Theory (can skip)
# The spin glass problem is reduced to the [cutting problem](@ref) for solving.
# Let ``G=(V,E)`` be a graph, the cutting problem can also be described by the following energy model
# ```math
# H^c = \sum_{ij \in E} C_{ij} (1 - n_i) n_j + (1 - n_j) n_i + \sum_{i \in V} w_i n_i,
# ```
# where ``n_i`` is the same as the partition index in the cutting problem,
# ``C_{ij} = -2J_{ij}`` are edge weights and ``w_i = 2h_i`` are vertex weights.
# The total energy is shifted by ``-\sum_{ij\in E}J_{ij} + \sum_{i \in V} h_i``.

# ## Solving properties
# ### Minimum and maximum energies
Emin = solve(problem, SizeMin())[]
#
Emax = solve(problem, SizeMax())[]

# ### Counting properties
# ##### graph polynomial
# The graph polynomial defined for the spin glass problem is a Laurent polynomial
# ```math
# Z(G, x) = \sum_{k=E_{\rm min}}^{E_{\rm max}(G)} c_k x^k,
# ```
# where ``E_{\rm min}`` and ``E_{\rm max}`` are minimum and maximum energies,
# ``c_k`` is the number of spin configurations with energy ``k``.
# Let ``x = e^\beta``, it corresponds to the partition function of a spin glass at temperature ``\beta^{-1}``.
partition_function = solve(problem, GraphPolynomial())[]

# ### Configuration properties
# ##### finding a ground state
ground_state = solve(problem, SingleConfigMin())[].c.data

Emin_verify = spinglass_energy(graph, ground_state)

# You should see a consistent result as above `Emin`.

show_graph(graph; locs=locations, vertex_colors=[
        iszero(ground_state[i]) ? "white" : "red" for i=1:nv(graph)], format=:svg)

# where a red vertice and a white vertice correspond to a spin having value 1 and 0 respectively.

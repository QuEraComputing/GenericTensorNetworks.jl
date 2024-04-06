# # Spin-glass problem

# !!! note
#     It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

using GenericTensorNetworks

## Spin-glass problem on a simple graph
# Let ``G=(V, E)`` be a graph, the [spin-glass](https://en.wikipedia.org/wiki/Spin_glass) problem in physics is characterized by the following energy function
# ```math
# H = \sum_{ij \in E} J_{ij} s_i s_j + \sum_{i \in V} h_i s_i,
# ```
# where ``h_i`` is an onsite energy term associated with spin ``s_i \in \{-1, 1\}``, and ``J_{ij}`` is the coupling strength between spins ``s_i`` and ``s_j``.
# In the program, we use boolean variable ``n_i = \frac{1-s_i}{2}`` to represent a spin configuration.

using Graphs

# In the following, we are going to defined an spin glass problem for the Petersen graph.

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 2.0, i) for i=0:4]..., [rot15(0.0, 1.0, i) for i=0:4]...]

show_graph(graph; locs=locations, format=:svg)

# ## Generic tensor network representation
# An anti-ferromagnetic spin glass problem can be defined with the [`SpinGlass`](@ref) type as
spinglass = SpinGlass(graph, fill(1, ne(graph)))

# The tensor network representation of the set packing problem can be obtained by
problem = GenericTensorNetwork(spinglass)

# ### Theory (can skip)
# We defined the reduction of the spin-glass problem to a tensor network on a hypergraph.
# Let ``H = (V, E)`` be a hypergraph. The tensor network for the partition function of the energy model for ``H`` can be defined as the following triple of (alphabet of labels, input tensors, output labels).
# ```math
# \begin{cases}
# \Lambda &= \{s_v \mid v \in V\}\\
# \mathcal{T} &= \{B^{(c)}_{s_{N(c, 1),N(c, 2),\ldots,N(c, d(c))}} \mid c \in E\} \cup \{W^{(v)}_{s_v} \mid v \in V\}\\
# \sigma_o &= \varepsilon
# \end{cases}
# ```
# where ``s_v \in \{0, 1\}`` is the boolean degreen associated to vertex ``v``,
# ``N(c, k)`` is the ``k``th vertex of hyperedge ``c``, and ``d(c)`` is the degree of ``c``.
# The edge tensor ``B^{(c)}`` is defined as
# ```math
# B^{(c)} = \begin{cases}
# x^{w_c} &  (\sum_{v\in c} s_v) \;is\; even, \\
# x^{-w_c}  &  otherwise.
# \end{cases}
# ```
# and the vertex tensor ``W^{(v)}`` (used to carry labels) is defined as
# ```math
# W^{(v)} = \left(\begin{matrix}1_v\\ 1_v\end{matrix}\right)
# ```

# ## Solving properties

# ### Minimum and maximum energies
# Its ground state energy is -9.
Emin = solve(problem, SizeMin())[]
# While the state correspond to the highest energy has the ferromagnetic order.
Emax = solve(problem, SizeMax())[]

# ### Counting properties
# ##### graph polynomial
# The graph polynomial defined for the spin glass problem is a Laurent polynomial
# ```math
# Z(G, J, h, x) = \sum_{k=E_{\rm min}}^{E_{\rm max}} c_k x^k,
# ```
# where ``E_{\rm min}`` and ``E_{\rm max}`` are minimum and maximum energies,
# ``c_k`` is the number of spin configurations with energy ``k``.
# Let ``x = e^\beta``, it corresponds to the partition function of a spin glass at temperature ``\beta^{-1}``.
partition_function = solve(problem, GraphPolynomial())[]

# ### Configuration properties
# ##### finding a ground state
ground_state = solve(problem, SingleConfigMin())[].c.data

Emin_verify = spinglass_energy(spinglass, ground_state)

# You should see a consistent result as above `Emin`.

show_graph(graph; locs=locations, vertex_colors=[
        iszero(ground_state[i]) ? "white" : "red" for i=1:nv(graph)], format=:svg)

# where a red vertex and a white vertex correspond to a spin having value 1 and 0 respectively.

# ## Spin-glass problem on a hypergraph
# A spin-glass problem on hypergraph ``H = (V, E)`` can be characterized by the following energy function
# ```math
# E = \sum_{c \in E} w_{c} \prod_{v\in c} S_v
# ```
# where ``S_v \in \{-1, 1\}``, ``w_c`` is coupling strength associated with hyperedge ``c``.
# In the program, we use boolean variable ``s_v = \frac{1-S_v}{2}`` to represent a spin configuration.

# In the following, we are going to defined an spin glass problem for the following hypergraph.
num_vertices = 15

hyperedges = [[1,3,4,6,7], [4,7,8,12], [2,5,9,11,13],
    [1,2,14,15], [3,6,10,12,14], [8,14,15], 
    [1,2,6,11], [1,2,4,6,8,12]]

weights = [-1, 1, -1, 1, -1, 1, -1, 1];

# The energy function is
# ```math
# \begin{align*}
# E = &-S_1S_3S_4S_6S_7 + S_4S_7S_8S_{12} - S_2S_5S_9S_{11}S_{13} +\\
#    &S_1S_2S_{14}S_{15} - S_3S_6S_{10}S_{12}S_{14} + S_8S_{14}S_{15} +\\
#    &S_1S_2S_6S_{11} - S_1s_2S_4S_6S_8S_{12}
# \end{align*}
# ```

# ## Generic tensor network representation
# We define an anti-ferromagnetic spin glass problem as
hyperspinglass = SpinGlass(num_vertices, hyperedges, weights);

# ## Solving properties
# We first define the problem as a tensor network.
hyperproblem = GenericTensorNetwork(hyperspinglass)

# ### Minimum and maximum energies
# Its ground state energy is -8.
Emin = solve(hyperproblem, SizeMin())[]
# While the state correspond to the highest energy has the ferromagnetic order.
Emax = solve(hyperproblem, SizeMax())[]

# In this example, the spin configurations can be chosen to make all hyperedges having even or odd spin parity.

# ### Counting properties
# ##### partition function and graph polynomial
# The graph polynomial defined for the spin-glass problem is a Laurent polynomial
# ```math
# Z(G, w, x) = \sum_{k=E_{\rm min}}^{E_{\rm max}} c_k x^k,
# ```
# where ``E_{\rm min}`` and ``E_{\rm max}`` are minimum and maximum energies,
# ``c_k`` is the number of spin configurations with energy ``k``.
# Let the inverse temperature ``\beta = 2``, the partition function is
β = 2.0
Z = solve(hyperproblem, PartitionFunction(β))[]

# The infinite temperature partition function is the counting of all feasible configurations
solve(hyperproblem, PartitionFunction(0.0))[]

# Let ``x = e^\beta``, it corresponds to the partition function of a spin glass at temperature ``\beta^{-1}``.
poly = solve(hyperproblem, GraphPolynomial())[]

# ### Configuration properties
# ##### finding a ground state
ground_state = solve(hyperproblem, SingleConfigMin())[].c.data

Emin_verify = spinglass_energy(hyperspinglass, ground_state)

# You should see a consistent result as above `Emin`.
"""
    GraphProblem

The abstract base type of graph problems.
"""
abstract type GraphProblem end

struct UnWeighted end

######## Interfaces for graph problems ##########
"""
    get_weights(problem::GraphProblem, sym)

The weight for `sym` of the graph problem, where `sym` is a symbol.
This interface is optional, the default weights are same as the flavors.
"""
get_weights(problem::GraphProblem, sym) = flavors(problem)

"""
    symbols(problem::GraphProblem)

The symbols of a graph problem, they are the degrees of freedoms in the graph.
"""
function symbols end

"""
    flavors(::Type{<:GraphProblem})

It returns a vector of integers as the flavors of a degree of freedom.
Its size is the same as the degree of freedom on a single vertex/edge.
"""
flavors(::GT) where GT<:GraphProblem = flavors(GT)

"""
    nflavor(::Type{<:GraphProblem})

Bond size is equal to the number of flavors.
"""
nflavor(::Type{GT}) where GT = length(flavors(GT))
nflavor(::GT) where GT<:GraphProblem = nflavor(GT)

"""
    generate_tensors(func, problem::GraphProblem)

Generate a vector of tensors as the inputs of the tensor network contraction code `problem.code`.
`func` is a function to customize the tensors.
`func(symbol)` returns a vector of elements, the length of which is same as the number of flavors.

Example
--------------------------
The following code gives your the maximum independent set size
```jldoctest
using Graph, GraphTensorNetworks
problem = Independence(smallgraph(:petersen))
f(x) = Tropical.([0, 1.0])
gp.code(generate_tensors(f, gp)...)
```
"""
generate_tensors(::Type{GT}) where GT = length(flavors(GT))

# requires field `code`

include("Independence.jl")
include("MaximalIndependence.jl")
include("MaxCut.jl")
include("Matching.jl")
include("Coloring.jl")
include("PaintShop.jl")

# forward the time, space and readwrite complexity
OMEinsum.timespacereadwrite_complexity(gp::GraphProblem) = timespacereadwrite_complexity(gp.code, uniformsize(gp.code, nflavor(gp)))

# contract the graph tensor network
function contractf(f, gp::GraphProblem; usecuda=false)
    @debug "generating tensors ..."
    xs = generate_tensors(f, gp)
    @debug "contracting tensors ..."
    if usecuda
        gp.code([CuArray(x) for x in xs]...)
    else
        gp.code(xs...)
    end
end

# TODOs:
# 1. Dominating set
# \exists x_i,\ldots,x_K \forall y\left[\bigwedge_{i=1}^{K}(y=x_i\wedge \textbf{adj}(y, x_i))\right]
# 2. Polish reading data
#     * consistent configuration assign of max-cut
# 3. Support transverse field in max-cut
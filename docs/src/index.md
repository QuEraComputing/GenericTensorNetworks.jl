```@meta
CurrentModule = GenericTensorNetworks
```

# GenericTensorNetworks

This package implements generic tensor networks to compute *solution space properties* of a class of hard combinatorial problems.
The *solution space properties* include
* The maximum/minimum solution sizes,
* The number of solutions at certain sizes,
* The enumeration of solutions at certain sizes.
* The direct sampling of solutions at certain sizes.

The solvable problems include [Independent set problem](@ref), [Maximal independent set problem](@ref), [Spin-glass problem](@ref), [Cutting problem](@ref), [Vertex matching problem](@ref), [Binary paint shop problem](@ref), [Coloring problem](@ref), [Dominating set problem](@ref), [Satisfiability problem](@ref), [Set packing problem](@ref) and [Set covering problem](@ref).

## Background knowledge

Please check our paper ["Computing properties of independent sets by generic programming tensor networks"](https://arxiv.org/abs/2205.03718).
If you find our paper or software useful in your work, we would be grateful if you could cite our work. The [CITATION.bib](https://github.com/QuEraComputing/GenericTensorNetworks.jl/blob/master/CITATION.bib) file in the root of this repository lists the relevant papers.

## Quick start

You can find a set up guide in our [README](https://github.com/QuEraComputing/GenericTensorNetworks.jl).
To get started, open a Julia REPL and type the following code.

```@repl
using GenericTensorNetworks, Graphs#, CUDA
solve(
           GenericTensorNetwork(IndependentSet(
                   Graphs.random_regular_graph(20, 3),
                   UnitWeight());    # default: uniform weight 1
               optimizer = TreeSA(),
               openvertices = (),       # default: no open vertices
               fixedvertices = Dict()   # default: no fixed vertices
           ),
           GraphPolynomial();
           usecuda=false              # the default value
       )
```

Here the main function [`solve`](@ref) takes three input arguments, the problem instance of type [`IndependentSet`](@ref), the property instance of type [`GraphPolynomial`](@ref) and an optional key word argument `usecuda` to decide use GPU or not.
If one wants to use GPU to accelerate the computation, the `, CUDA` should be uncommented.

An [`IndependentSet`](@ref) instance takes two positional arguments to initialize, the graph instance that one wants to solve and the get_weights for each vertex. Here, we use a random regular graph with 20 vertices and degree 3, and the default uniform weight 1.

The [`GenericTensorNetwork`](@ref) function is a constructor for the problem instance, which takes the problem instance as the first argument and optional key word arguments. The key word argument `optimizer` is for specifying the tensor network optimization algorithm.
The keyword argument `openvertices` is a tuple of labels for specifying the degrees of freedom not summed over, and `fixedvertices` is a label-value dictionary for specifying the fixed values of the degree of freedoms.
Here, we use [`TreeSA`](@ref) method as the tensor network optimizer, and leave `openvertices` the default values.
The [`TreeSA`](@ref) method finds the best contraction order in most of our applications, while the default [`GreedyMethod`](@ref) runs the fastest.

The first execution of this function will be a bit slow due to Julia's just in time compiling.
The subsequent runs will be fast.
The following diagram lists possible combinations of input arguments, where functions in the `Graph` are mainly defined in the package [Graphs](https://github.com/JuliaGraphs/Graphs.jl), and the rest can be found in this package.
```@raw html
<div align=center>
<img src="assets/fig7.svg" width="75%"/>
</div>
```⠀
You can find many examples in this documentation, a good one to start with is [Independent set problem](@ref).


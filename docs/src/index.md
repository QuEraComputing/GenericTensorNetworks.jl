```@meta
CurrentModule = GenericTensorNetworks
```

# GenericTensorNetworks

## Overview
GenericTensorNetworks is a high-performance package that uses tensor network algorithms to solve challenging combinatorial optimization problems. This approach allows us to efficiently compute various solution space properties that would be intractable with traditional methods.

## Key Capabilities
Our package can compute a wide range of solution space properties:

* Maximum and minimum solution sizes
* Solution counts at specific sizes
* Complete enumeration of solutions
* Statistical sampling from the solution space

## Supported Problem Classes
GenericTensorNetworks can solve many important combinatorial problems:

* [Independent Set Problem](@ref)
* [Maximal Independent Set Problem](@ref)
* [Spin-Glass Problem](@ref)
* [Maximum Cut Problem](@ref)
* [Vertex Matching Problem](@ref)
* [Binary Paint Shop Problem](@ref)
* [Graph Coloring Problem](@ref)
* [Dominating Set Problem](@ref)
* [Boolean Satisfiability Problem](@ref)
* [Set Packing Problem](@ref)
* [Set Covering Problem](@ref)

## Scientific Background
For the theoretical foundation and algorithmic details, please refer to our paper:
["Computing properties of independent sets by generic programming tensor networks"](https://arxiv.org/abs/2205.03718)

If you find our package useful in your research, please cite our work using the references in [CITATION.bib](https://github.com/QuEraComputing/GenericTensorNetworks.jl/blob/master/CITATION.bib).

## Getting Started

### Installation
Installation instructions are available in our [README](https://github.com/QuEraComputing/GenericTensorNetworks.jl).

### Basic Example
Here's a simple example that computes the independence polynomial of a random regular graph:

```julia
using GenericTensorNetworks, Graphs  # Add CUDA for GPU acceleration

# Create and solve a problem instance
result = solve(
    GenericTensorNetwork(
        IndependentSet(
            Graphs.random_regular_graph(20, 3),  # Graph to analyze
            UnitWeight(20)                       # Uniform vertex weights
        );
        optimizer = TreeSA(),                    # Contraction order optimizer
        openvertices = (),                       # No open vertices
        fixedvertices = Dict()                   # No fixed vertices
    ),
    GraphPolynomial();                           # Property to compute
    usecuda = false                              # Use CPU (set true for GPU)
)
```

### Understanding the API

The main function `solve` takes three components:

1. **Problem Instance**: Created with `GenericTensorNetwork`, which wraps problem types like `IndependentSet`
   - The first argument defines the problem (graph and weights)
   - Optional arguments control the tensor network construction:
     - `optimizer`: Algorithm for finding efficient contraction orders
     - `openvertices`: Degrees of freedom to leave uncontracted
     - `fixedvertices`: Variables with fixed assignments

2. **Property to Compute**: Such as `GraphPolynomial`, `SizeMax`, or `ConfigsAll`

3. **Computation Options**: Like `usecuda` to enable GPU acceleration

Note: The first execution will be slower due to Julia's just-in-time compilation. Subsequent runs will be much faster.

### API Structure
The following diagram illustrates the possible combinations of inputs:

```@raw html
<div align=center>
<img src="assets/fig7.svg" width="75%"/>
</div>
```⠀

Functions in the `Graph` box are primarily from the [Graphs](https://github.com/JuliaGraphs/Graphs.jl) package, while the rest are defined in GenericTensorNetworks.

## Next Steps
For a deeper understanding, we recommend starting with the [Independent Set Problem](@ref) example, which demonstrates the core functionality of the package.
```
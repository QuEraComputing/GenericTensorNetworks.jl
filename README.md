# GenericTensorNetworks

[![CI](https://github.com/QuEraComputing/GenericTensorNetworks.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/QuEraComputing/GenericTensorNetworks.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/QuEraComputing/GenericTensorNetworks.jl/branch/master/graph/badge.svg?token=vwWQntOxvG)](https://codecov.io/gh/QuEraComputing/GenericTensorNetworks.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/)


This package implements generic tensor networks to compute *solution space properties* of a class of hard combinatorial optimization problems.
The *solution space properties* include
* The maximum/minimum solution sizes,
* The number of solutions at certain sizes,
* The enumeration/sampling of solutions at certain sizes.

The types of problems that can be solved using this package include [Independent set problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/IndependentSet/), [Maximal independent set problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/MaximalIS/), [Spin-glass problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/SpinGlass/), [Cutting problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/MaxCut/), [Vertex matching problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/Matching/), [Binary paint shop problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/PaintShop/), [Coloring problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/Coloring/), [Dominating set problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/DominatingSet/), [Set packing problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/SetPacking/), [Satisfiability problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/Satisfiability/) and [Set covering problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/SetCovering/).

## Installation
<p>
GenericTensorNetworks is a &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia Language
    </a>
    &nbsp; package. To install GenericTensorNetworks,
    please <a href="https://docs.julialang.org/en/v1/manual/getting-started/">open
    Julia's interactive session (known as REPL)</a> and press the <kbd>]</kbd> key in the REPL to use the package mode, and then type:
</p>

```julia
pkg> add GenericTensorNetworks
```

To update, just type `up` in the package mode.

We recommend that you use **Julia version >= 1.7**; otherwise, your program may suffer from significant (exponential in the tensor dimension) overheads when permuting the dimensions of a large tensor.
If you have to use an older version of Julia, you can overwrite the `LinearAlgebra.permutedims!` by adding the following patch to your own project.

```julia
# only required when your Julia version is < 1.7
using TensorOperations, LinearAlgebra
function LinearAlgebra.permutedims!(C::Array{T,N}, A::StridedArray{T,N}, perm) where {T,N}
    if isbitstype(T)
        TensorOperations.tensorcopy!(A, ntuple(identity,N), C, perm)
    else
        invoke(permutedims!, Tuple{Any,AbstractArray,Any}, C, A, perm)
    end
end
```

## Supporting and Citing

Much of the software in this ecosystem was developed as a part of an academic research project.
If you would like to help support it, please star the repository.
If you use our software as part of your research, teaching, or other activities, we would like to request you to cite our [work](https://arxiv.org/abs/2205.03718). 
The
[CITATION.bib](https://github.com/QuEraComputing/GenericTensorNetworks.jl/blob/master/CITATION.bib) file in the root of this repository lists the relevant papers.

## Questions and Contributions

You can
* Post a question on [Julia Discourse forum](https://discourse.julialang.org/) and ping the package maintainer with `@1115`.
* Discuss in the `#graphs` channel of the [Julia Slack](https://julialang.org/community/) and ping the package maintainer with `@JinGuo Liu`.
* Open an [issue](https://github.com/QuEraComputing/GenericTensorNetworks.jl/issues) if you encounter any problems, or have any feature request.

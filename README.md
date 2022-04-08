# GraphTensorNetworks

[![Build Status](https://github.com/QuEraComputing/GraphTensorNetworks.jl/workflows/CI/badge.svg)](https://github.com/QuEraComputing/GraphTensorNetworks.jl/actions)
[![Coverage Status](https://coveralls.io/repos/github/QuEraComputing/GraphTensorNetworks.jl/badge.svg?branch=master&t=rIJIK2)](https://coveralls.io/github/QuEraComputing/GraphTensorNetworks.jl?branch=master)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://psychic-meme-f4d866f8.pages.github.io/dev/)

## Installation
<p>
GraphTensorNetworks is a &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia Language
    </a>
    &nbsp; package. To install GraphTensorNetworks,
    please <a href="https://docs.julialang.org/en/v1/manual/getting-started/">open
    Julia's interactive session (known as REPL)</a> and press <kbd>]</kbd> key in the REPL to use the package mode, then type
</p>

```julia
pkg> add GraphTensorNetworks
```

To update, just type `up` in the package mode.

We recommend using **Julia version >= 1.7**, otherwise your program can suffer from significant (exponential in tensor dimension) overheads when permuting the dimensions of a large tensor.
If you have to use an older version Julia, you can overwrite the `LinearAlgebra.permutedims!` by adding the following patch to your own project.

```julia
# only required when your Julia version < 1.7
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

Much of the software in this ecosystem was developed as part of academic research.
If you would like to help support it, please star the repository as such metrics may help us secure funding in the future.
If you use our software as part of your research, teaching, or other activities, we would be grateful if you could cite our work.
The
[CITATION.bib](https://github.com/QuEraComputing/GraphTensorNetworks.jl/blob/master/CITATION.bib) file in the root of this repository lists the relevant papers.

## Questions and Contributions

You can
* Post a question on [Julia Discourse forum](https://discourse.julialang.org/), pin the package maintainer wih `@1115`.
* Discuss in the `#graphs` channel of the [Julia Slack](https://julialang.org/community/), ping the package maintainer with `@JinGuo Liu`.
* Open an [issue](https://github.com/QuEraComputing/GraphTensorNetworks.jl/issues) if you encounter any problems, or have any feature request.

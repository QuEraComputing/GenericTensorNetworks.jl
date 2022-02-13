# GraphTensorNetworks

[![Build Status](https://github.com/Happy-Diode/GraphTensorNetworks.jl/workflows/CI/badge.svg)](https://github.com/Happy-Diode/GraphTensorNetworks.jl/actions)
[![Coverage Status](https://coveralls.io/repos/github/Happy-Diode/GraphTensorNetworks.jl/badge.svg?branch=master&t=rIJIK2)](https://coveralls.io/github/Happy-Diode/GraphTensorNetworks.jl?branch=master)
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
    Julia's interactive session (known as REPL)</a> and press <kbd>]</kbd> key in the REPL to use the package mode, then
</p>

1. if you are a user, just type
```julia
pkg> add GraphTensorNetworks
```

If you do not have access to our registry, e.g. you are an external collaborator, you can install the master branch by typing
```julia
pkg> add https://github.com/Happy-Diode/GraphTensorNetworks.jl.git#master
```

To update, just type `up` in the package mode.

2. If you are a developer, you can install it in develop mode
```julia
pkg> dev https://github.com/Happy-Diode/GraphTensorNetworks.jl.git
```

Packages installed in developer mode will not be updated by the `up` command, you should go to the develop folder and use `git` to manage your versions. For more [details](https://docs.julialang.org/en/v1/stdlib/Pkg/).

Please use **Julia version >= 1.7**, otherwise you will suffer from huge overheads when contracting large tensor networks. If you have to use an old version Julia,
you can avoid the overhead by overriding the `permutedims!` is `LinearAlgebra`, i.e. add the following code to your own project.

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

## Examples

In this example, we will show how to compute the independent set properties of the Petersen graph, we first generate its tensor network contraction tree.
```julia
julia> using GraphTensorNetworks, Random, Graphs

julia> graph = (Random.seed!(2); Graphs.smallgraph(:petersen))
{10, 15} undirected simple Int64 graph

julia> problem = IndependentSet(graph);
```

Here, the `problem` is a `IndependentSet` instance, it contains the tensor network contraction tree for the target graph (the `code` field).

#### 1. find MIS size, count MISs and count ISs
* maximum independent set size
```julia
julia> solve(problem, SizeMax())[]
4.0ₜ
```
Here, the `solve` function returns you a 0-dimensional array.
For open graphs, this output tensor can have nonzero dimensionality. Each entry corresponds to a different boundary condition.

* all independent sets
```julia
julia> solve(problem, CountingAll())[]
76.0
```

* counting maximum independent sets
```julia
julia> solve(problem, CountingMax())[]
(4.0, 5.0)ₜ  # first field is MIS size, second is its counting.
```

* counting independent sets of max two sizes with truncated polynomial
```julia
julia> solve(problem, CountingMax(2))[]
0-dimensional Array{Max2Poly{Float64, Float64}, 0}:
30.0*x^3 + 5.0*x^4
```

The following code computes independence polynomial using the finite field algebra (default) approach.
It is equivalent to counting independent sets of an arbituary size.

```julia
julia> solve(problem, GraphPolynomial())[]
Polynomial(1 + 10*x + 30*x^2 + 30*x^3 + 5*x^4)
```

The program use `finitefield` method as the default approach, because it has no round off error is can be upload to GPU.

#### 3. find/enumerate solutions
* find one of the best solutions,
```julia
julia> solve(problem, SingleConfigMax())[]
(4.0, ConfigSampler{10, 1, 1}(1010000011))ₜ
```

* enumerate all MISs
```julia
julia> cs = solve(problem, ConfigsMax())[]
0-dimensional Array{CountingTropical{Int64, ConfigEnumerator{10, 1, 1}}, 0}:
(4, {1010000011, 0100100110, 1001001100, 0010111000, 0101010001})ₜ
```
It will use the bounded version to save the computational effort.  If you want to save/load your configurations, you can type
```julia
julia> save_configs("configs.dat", cs.c; format=:text)  # `:text` or `:binary`

julia> load_configs("configs.dat"; format=:text)
{1010000011, 0100100110, 1001001100, 0010111000, 0101010001}
```

* enumerate all configurations of size α(G) and α(G)-1
```julia
julia> solve(problem, ConfigsMax(2))[]
{0010101000, 0101000001, 0100100010, 0010100010, 0100000011, 0010000011, 1001001000, 1010001000, 1001000001, 1010000001, 1010000010, 1000000011, 0100100100, 0000101100, 0101000100, 0001001100, 0000100110, 0100000110, 1001000100, 1000001100, 1000000110, 0100110000, 0000111000, 0101010000, 0001011000, 0010110000, 0010011000, 0001010001, 0100010001, 0010010001}*x^3 + {1010000011, 0100100110, 1001001100, 0010111000, 0101010001}*x^4
```

* enumerate all independent sets
```julia
julia> solve(problem, ConfigsAll())[]
{0000000000, 0000010000, 1000000000, 0001000000, 0001010000, 1001000000, 0010000000, 0010010000, 1010000000, 0000001000, 0000011000, 1000001000, 0001001000, 0001011000, 1001001000, 0010001000, 0010011000, 1010001000, 0000000010, 1000000010, 0010000010, 1010000010, 0100000000, 0100010000, 0101000000, 0101010000, 0100000010, 0000000100, 1000000100, 0001000100, 1001000100, 0000001100, 1000001100, 0001001100, 1001001100, 0000000110, 1000000110, 0100000100, 0101000100, 0100000110, 0000100000, 0000110000, 0010100000, 0010110000, 0000101000, 0000111000, 0010101000, 0010111000, 0000100010, 0010100010, 0100100000, 0100110000, 0100100010, 0000100100, 0000101100, 0000100110, 0100100100, 0100100110, 0000000001, 0000010001, 1000000001, 0001000001, 0001010001, 1001000001, 0010000001, 0010010001, 1010000001, 0000000011, 1000000011, 0010000011, 1010000011, 0100000001, 0100010001, 0101000001, 0101010001, 0100000011}
```

## Supporting and Citing

Much of the software in this ecosystem was developed as part of academic research. If you
would like to help support it, please star the repository as such metrics may help us secure
funding in the future. If you use our software as part of your research, teaching, or other
activities, we would be grateful if you could cite our work. The
[CITATION.bib](https://github.com/Happy-Diode/GraphTensorNetworks.jl/blob/master/CITATION.bib) file in the root of this repository lists the relevant papers.

## Questions and Contributions

You can
* Post a question on [Julia Discourse forum](https://discourse.julialang.org/), pin the package maintainer wih `@1115`.
* Discuss in the `#graphs` channel of the [Julia Slack](https://julialang.org/community/), ping the package maintainer with `@JinGuo Liu`.
* Open an [issue](https://github.com/Happy-Diode/GraphTensorNetworks.jl/issues) if you encounter any problems, or have any feature request.
# GraphTensorNetworks

[![Build Status](https://github.com/Happy-Diode/GraphTensorNetworks.jl/workflows/CI/badge.svg)](https://github.com/Happy-Diode/GraphTensorNetworks.jl/actions)

## Installation
<p>
GraphTensorNetworks is a &nbsp;
    <a href="https://julialang.org">
        <img src="https://julialang.org/favicon.ico" width="16em">
        Julia Language
    </a>
    &nbsp; package. To install GraphTensorNetworks,
    please <a href="https://docs.julialang.org/en/v1/manual/getting-started/">open
    Julia's interactive session (known as REPL)</a> and press <kbd>]</kbd> key in the REPL to use the package mode, then type the following command
</p>

```julia
pkg> add GraphTensorNetworks
```

Please use Julia-1.7, otherwise you will suffer from huge overheads when contracting large tensor networks. If you have to use a lower version,
you can avoid the overhead by overriding the `permutedims!` is `LinearAlgebra`.

```julia
using TensorOperations, LinearAlgebra
function LinearAlgebra.permutedims!(C::Array{T,N}, A::StridedArray{T,N}, perm) where {T,N}
    if isbitstype(T)
        TensorOperations.tensorcopy!(A, ntuple(identity,N), C, perm)
    else
        invoke(permutedims!, Tuple{Any,AbstractArray,Any}, C, A, perm)
    end
end
```
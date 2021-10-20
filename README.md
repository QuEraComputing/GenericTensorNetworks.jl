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

Please use **Julia-1.7**, otherwise you will suffer from huge overheads when contracting large tensor networks. If you have to use a lower version,
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

## Examples

Let us use the Petersen graph as an example, we first generate its tensor network contraction tree.
```julia
julia> using GraphTensorNetworks, Random, LightGraphs

julia> graph = (Random.seed!(2); LightGraphs.smallgraph(:petersen))
{10, 15} undirected simple Int64 graph

julia> problem = Independence(graph; optimizer=TreeSA(sc_target=0, sc_weight=1.0, ntrials=10, βs=0.01:0.1:15.0, niters=20, rw_weight=0.2));
┌ Warning: target space complexity not found, got: 4.0, with time complexity 7.965784284662087, read-right complexity 8.661778097771988.
└ @ OMEinsumContractionOrders ~/.julia/dev/OMEinsumContractionOrders/src/treesa.jl:71
time/space complexity is (7.965784284662086, 4.0)
```

Here, the `problem` is a `Independence` instance, it contains the tensor network contraction tree for the target graph.
Here, we choose the `:tree` optimizer to optimize the tensor network contraciton tree, it is a local search based algorithm, check [arXiv: 2108.05665](https://arxiv.org/abs/2108.05665). You will see some warnings, do not panic, this is because we set `sc_target` (target space complex) to 1 for agressive optimization of space complexity. Type `?Independence` in a Julia REPL for more information about the key word arguments.
Similarly, one can select tensor network structures for solving other problems like `MaximalIndependence`, `MaxCut`, `Matching`, `Coloring{K}` and `set_packing`.

#### 1. find MIS size, count MISs and count ISs
```julia
# maximum independent set size
julia> solve(problem, "size max")
0-dimensional Array{TropicalNumbers.TropicalF64, 0}:
4.0ₜ

# all independent sets
julia> solve(problem, "counting sum")
0-dimensional Array{Float64, 0}:
76.0

# counting maximum independent sets
julia> solve(problem, "counting max")
0-dimensional Array{TropicalNumbers.CountingTropicalF64, 0}:
(4.0, 5.0)ₜ

# counting independent sets of max two sizes
julia> solve(problem, "counting max2")
0-dimensional Array{Max2Poly{Float64, Float64}, 0}:
30.0*x^3 + 5.0*x^4
```

Here, `solve` function returns you a 0-dimensional array.
For open graphs, this output tensor can have higher dimensions. Each entry corresponds to a different boundary condition.

#### 2. compute the independence polynomial

```julia
# using `Polynomial` type
julia> solve(problem, "counting all")
0-dimensional Array{Polynomial{Float64, :x}, 0}:
Polynomial(1.0 + 10.0*x + 30.0*x^2 + 30.0*x^3 + 5.0*x^4)

# using the finitefield approach
julia> solve(problem, "counting all (finitefield)")
0-dimensional Array{Polynomial{BigInt, :x}, 0}:
Polynomial(1 + 10*x + 30*x^2 + 30*x^3 + 5*x^4)

# using the fourier approach
julia> solve(problem, "counting all (fft)", r=1.0)
0-dimensional Array{Polynomial{ComplexF64, :x}, 0}:
Polynomial(1.0000000000000029 + 2.664535259100376e-16im + (10.000000000000004 - 1.9512435398857492e-16im)x + (30.0 - 1.9622216671393801e-16im)x^2 + (30.0 + 1.1553104311877194e-15im)x^3 + (5.0 - 1.030417436395244e-15im)x^4)
```

The `finitefield` approach is the most recommended, because it has no round off error is can be upload to GPU. To upload the computing to GPU,
```julia
julia> using CUDA
[ Info: OMEinsum loaded the CUDA module successfully

julia> solve(problem, "counting all (finitefield)", usecuda=true)
0-dimensional Array{Polynomial{BigInt, :x}, 0}:
Polynomial(1 + 10*x + 30*x^2 + 30*x^3 + 5*x^4)
```

The `fft` approach is fast but with round off errors. Its imaginary part can be regarded as the precision,
keyword argument `r` controls the round off errors in high/low IS size region.

#### 3. find/enumerate solutions
```julia
# one of MISs
julia> solve(problem, "config max")
0-dimensional Array{CountingTropical{Float64, ConfigSampler{10, 1, 1}}, 0}:
(4.0, ConfigSampler{10, 1, 1}(1010000011))ₜ

julia> solve(problem, "config max (bounded)")
0-dimensional Array{CountingTropical{Float64, ConfigSampler{10, 1, 1}}, 0}:
(4.0, ConfigSampler{10, 1, 1}(1010000011))ₜ

# enumerate all MISs
julia> solve(problem, "configs max")  # not recommended
0-dimensional Array{CountingTropical{Float64, ConfigEnumerator{10, 1, 1}}, 0}:
(4.0, {1010000011, 0100100110, 1001001100, 0010111000, 0101010001})ₜ

julia> solve(problem, "configs max (bounded)")
0-dimensional Array{CountingTropical{Int64, ConfigEnumerator{10, 1, 1}}, 0}:
(4, {1010000011, 0100100110, 1001001100, 0010111000, 0101010001})ₜ

# enumerate all MIS and MIS-1 configurations
julia> solve(problem, "configs max2")
0-dimensional Array{Max2Poly{ConfigEnumerator{10, 1, 1}, Float64}, 0}:
{0010101000, 0101000001, 0100100010, 0010100010, 0100000011, 0010000011, 1001001000, 1010001000, 1001000001, 1010000001, 1010000010, 1000000011, 0100100100, 0000101100, 0101000100, 0001001100, 0000100110, 0100000110, 1001000100, 1000001100, 1000000110, 0100110000, 0000111000, 0101010000, 0001011000, 0010110000, 0010011000, 0001010001, 0100010001, 0010010001}*x^3 + {1010000011, 0100100110, 1001001100, 0010111000, 0101010001}*x^4

# enumerate all IS configurations
julia> solve(problem, "configs all")
0-dimensional Array{Polynomial{ConfigEnumerator{10, 1, 1}, :x}, 0}:
Polynomial({0000000000} + {0010000000, 0000100000, 0001000000, 0100000000, 0000001000, 0000000001, 0000000010, 1000000000, 0000000100, 0000010000}*x + {1000000010, 0010100000, 0010001000, 0100100000, 0000101000, 0101000000, 0001001000, 0001000001, 0100000001, 0010000001, 0000100010, 0100000010, 0010000010, 0000000011, 1001000000, 1000001000, 1010000000, 1000000001, 0000000110, 0000100100, 0001000100, 0100000100, 0000001100, 1000000100, 0010010000, 0000110000, 0001010000, 0100010000, 0000011000, 0000010001}*x^2 + {1010000010, 1000000011, 0010101000, 0101000001, 0100100010, 0010100010, 0100000011, 0010000011, 1001001000, 1010001000, 1001000001, 1010000001, 0000100110, 0100000110, 0100100100, 0000101100, 0101000100, 0001001100, 1001000100, 1000001100, 1000000110, 0010110000, 0010011000, 0100110000, 0000111000, 0101010000, 0001011000, 0001010001, 0100010001, 0010010001}*x^3 + {1010000011, 0100100110, 1001001100, 0010111000, 0101010001}*x^4)
```

If you want to enumerate all MISs, we highly recommend using the bounded version to save the computational effort. One can also store the configurations on your disk by typing
```julia
julia> cs = solve(problem, "configs max (bounded)")[1].c  # the `c` field is a `ConfigEnumerator`
{1010000011, 0100100110, 1001001100, 0010111000, 0101010001}

julia> save_configs("configs.dat", cs; format=:text)  # `:text` or `:binary`

julia> load_configs("configs.dat"; format=:text)
{1010000011, 0100100110, 1001001100, 0010111000, 0101010001}
```

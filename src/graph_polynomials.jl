using Polynomials
using OMEinsum: NestedEinsum, getixs, getiy
using FFTW
using LightGraphs

export contractx, graph_polynomial, contraction_code
export Independence, MaximalIndependence, Matching, Coloring
const EinTypes = Union{EinCode,NestedEinsum}

struct Independence end
struct MaximalIndependence end
struct Matching end
struct Coloring end

function graph_polynomial(which, approach::Val, g::SimpleGraph; method=:kahypar, sc_target=17, max_group_size=40, nrepeat=10, imbalances=0.0:0.01:0.2, kwargs...)
    code = contraction_code(which, g; method=method, sc_target=sc_target, max_group_size=max_group_size, nrepeat=nrepeat, imbalances=imbalances)
    graph_polynomial(which, approach, code; kwargs...)
end

function graph_polynomial(which, ::Val{:fft}, code::EinTypes; usecuda=false, maxorder=graph_polynomial_maxorder(which, code; usecuda=usecuda), r=1.0)
	ω = exp(-2im*π/(maxorder+1))
	xs = r .* collect(ω .^ (0:maxorder))
	ys = [asscalar(contractx(which, x, code; usecuda=usecuda)) for x in xs]
	Polynomial(ifft(ys) ./ (r .^ (0:maxorder)))
end

function graph_polynomial(which, ::Val{:fitting}, code::EinTypes; usecuda=false,
        maxorder = graph_polynomial_maxorder(which, code; usecuda=usecuda))
	xs = (0:maxorder)
	ys = [asscalar(contractx(which, x, code; usecuda=usecuda)) for x in xs]
	fit(xs, ys, maxorder)
end

function graph_polynomial(which, ::Val{:polynomial}, code::EinTypes; usecuda=false)
    @assert !usecuda "Polynomial type can not be computed on GPU!"
    contractx(which, Polynomial([0, 1.0]), code)
end

function _polynomial_single(which, ::Type{T}, code::EinTypes; usecuda, maxorder) where T
	xs = 0:maxorder
	ys = [asscalar(contractx(which, T(x), code; usecuda=usecuda)) for x in xs]
	A = zeros(T, maxorder+1, maxorder+1)
	for j=1:maxorder+1, i=1:maxorder+1
		A[j,i] = T(xs[j])^(i-1)
	end
	A \ T.(ys)
end

function graph_polynomial(which, ::Val{:finitefield}, code::EinTypes; usecuda=false, maxorder=graph_polynomial_maxorder(which, code; usecuda=usecuda), max_iter=100)
    TI = Int32  # Int 32 is faster
    N = typemax(TI)
    YS = []
    local res
    for k = 1:max_iter
	    N = prevprime(N-TI(1))
        T = Mods.Mod{N,TI}
        rk = _polynomial_single(which, T, code; usecuda=usecuda, maxorder=maxorder)
        push!(YS, rk)
        if maxorder==1
            return Polynomial(Mods.value.(YS[1]))
        elseif k != 1
            ra = improved_counting(YS[1:end-1])
            res = improved_counting(YS)
            ra == res && return Polynomial(res)
        end
    end
    @warn "result is potentially inconsistent."
    return Polynomial(res)
end
function improved_counting(sequences)
    map(yi->Mods.CRT(yi...), zip(sequences...))
end

function contraction_code(which, g::SimpleGraph; method=:kahypar, sc_target=17, max_group_size=40, nrepeat=10, imbalances=0.0:0.001:0.8)
    _optimize_code(_code(which, g), method, sc_target, max_group_size, nrepeat, imbalances)
end
function _optimize_code(code, method, sc_target, max_group_size, nrepeat, imbalances)
    size_dict = Dict([s=>2 for s in symbols(code)])
    optcode = if method == :kahypar
        optimize_kahypar(code, size_dict; sc_target=sc_target, max_group_size=max_group_size, imbalances=imbalances)
    elseif method == :greedy
        optimize_greedy(code, size_dict; nrepeat=nrepeat)
    else
        ArgumentError("optimizer `$method` not defined.")
    end
    println("time/space complexity is $(OMEinsum.timespace_complexity(optcode, size_dict))")
    return optcode
end

############### Problem specific implementations ################
### independent set ###
function _code(::Independence, g::SimpleGraph)
    EinCode(([(i,) for i in LightGraphs.vertices(g)]..., # labels for edge tensors
                    [minmax(e.src,e.dst) for e in LightGraphs.edges(g)]...), ())        # labels for vertex tensors
end

function contractx(::Independence, x::T, code::EinTypes; usecuda=false) where {T}
    tensors = map(getixs(flatten(code))) do ix
        # if the tensor rank is 1, create a vertex tensor.
        # otherwise the tensor rank must be 2, create a bond tensor.
        t = length(ix)==1 ? misv(T, x) : misb(T)
        usecuda ? CuArray(t) : t
    end
	dynamic_einsum(code, tensors)
end
misb(::Type{T}) where T = [one(T) one(T); one(T) zero(T)]
misv(::Type{T}, val) where T = [one(T), convert(T, val)]

graph_polynomial_maxorder(::Independence, code; usecuda) = Int(sum(contractx(Independence(), TropicalF64(1.0), code; usecuda=usecuda)).n)

### coloring ###
_code(::Coloring, g::SimpleGraph) = independence_code(args...; kwargs...)
function contractx(::Coloring, xs, code::EinTypes; usecuda=false)
    tensors = map(getixs(flatten(code))) do ix
        # if the tensor rank is 1, create a vertex tensor.
        # otherwise the tensor rank must be 2, create a bond tensor.
        t = length(ix)==1 ? coloringv(collect(xs)) : coloringb(eltype(xs), length(xs))
        usecuda ? CuArray(t) : t
    end
	dynamic_einsum(code, tensors)
end

# coloring bond tensor
function coloringb(::Type{T}, k::Int) where T
    x = ones(T, k, k)
    for i=1:k
        x[i,i] = zero(T)
    end
    return x
end
# coloring vertex tensor
coloringv(vals::Vector{T}) where T = vals

### matching ###
function _code(::Matching, g::SimpleGraph)
    EinCode(([(minmax(e.src,e.dst),) for e in LightGraphs.edges(g)]..., # labels for edge tensors
                    [([minmax(i,j) for j in neighbors(g, i)]...,) for i in LightGraphs.vertices(g)]...,), ())        # labels for vertex tensors
end

function contractx(::Matching, x::T, optcode::EinTypes; usecuda=false) where T
    ixs = OMEinsum.getixs(flatten(optcode))
    n = length(unique(Iterators.flatten(ixs)))  # number of vertices
    tensors = []
    for i=1:length(ixs)
        if i<=n
            @assert length(ixs[i]) == 1
            t = T[one(T), x]
        else
            t = match_tensor(T, length(ixs[i]))
        end
        push!(tensors, usecuda ? CuArray(t) : t)
    end
	optcode(tensors...)
end
function match_tensor(::Type{T}, n::Int) where T
    t = zeros(T, fill(2, n)...)
    for ci in CartesianIndices(t)
        if sum(ci.I .- 1) <= 1
            t[ci] = one(T)
        end
    end
    return t
end

graph_polynomial_maxorder(::Matching, code; usecuda) = Int(sum(contractx(Matching(), TropicalF64(1.0), code; usecuda=usecuda)).n)

### maximal independent set ###
function _code(::MaximalIndependence, g::SimpleGraph)
    EinCode(([(LightGraphs.neighbors(g, v)..., v) for v in LightGraphs.vertices(g)]...,), ())
end

function contractx(::MaximalIndependence, x::T, optcode::EinTypes; usecuda=false) where T
    ixs = OMEinsum.getixs(flatten(optcode))
	tensors = map(ixs) do ix
        t = neighbortensor(x, length(ix))
        usecuda ? CuArray(t) : t
    end
	dynamic_einsum(optcode, tensors)
end
function neighbortensor(x::T, d::Int) where T
    t = zeros(T, fill(2, d)...)
    for i = 2:1<<(d-1)
        t[i] = one(T)
    end
    t[1<<(d-1)+1] = x
    return t
end

graph_polynomial_maxorder(::MaximalIndependence, code; usecuda) = Int(sum(contractx(MaximalIndependence(), TropicalF64(1.0), code; usecuda=usecuda)).n)
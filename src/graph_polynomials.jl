using Polynomials
using OMEinsum: NestedEinsum, getixs, getiy
using FFTW
using LightGraphs

export contractx, graph_polynomial, optimize_code
export Independence, MaximalIndependence, Matching, Coloring
const EinTypes = Union{EinCode,NestedEinsum}

abstract type GraphProblem end
struct Independence{CT<:EinTypes} <: GraphProblem
    code::CT
end
struct MaximalIndependence{CT<:EinTypes} <: GraphProblem
    code::CT
end
struct Matching{CT<:EinTypes} <: GraphProblem
    code::CT
end
struct Coloring{K,CT<:EinTypes} <: GraphProblem
    code::CT
end
Coloring{K}(code::ET) where {K,ET<:EinTypes} = Coloring{K,ET}(code)

"""
    labels(code)

Return a vector of unique labels in an Einsum token.
"""
function labels(code::EinTypes)
    res = []
    for ix in OMEinsum.getixs(OMEinsum.flatten(code))
        for l in ix
            if l ∉ res
                push!(res, l)
            end
        end
    end
    return res
end

function graph_polynomial(gp::GraphProblem, ::Val{:fft}; usecuda=false, maxorder=graph_polynomial_maxorder(gp; usecuda=usecuda), r=1.0)
	ω = exp(-2im*π/(maxorder+1))
	xs = r .* collect(ω .^ (0:maxorder))
	ys = [asscalar(contractx(gp, x; usecuda=usecuda)) for x in xs]
	Polynomial(ifft(ys) ./ (r .^ (0:maxorder)))
end

function graph_polynomial(gp::GraphProblem, ::Val{:fitting}; usecuda=false,
        maxorder = graph_polynomial_maxorder(gp; usecuda=usecuda))
	xs = (0:maxorder)
	ys = [asscalar(contractx(gp, x; usecuda=usecuda)) for x in xs]
	fit(xs, ys, maxorder)
end

function graph_polynomial(gp::GraphProblem, ::Val{:polynomial}; usecuda=false)
    @assert !usecuda "Polynomial type can not be computed on GPU!"
    contractx(gp::GraphProblem, Polynomial([0, 1.0]))
end

function _polynomial_single(gp::GraphProblem, ::Type{T}; usecuda, maxorder) where T
	xs = 0:maxorder
	ys = [asscalar(contractx(gp, T(x); usecuda=usecuda)) for x in xs]
	A = zeros(T, maxorder+1, maxorder+1)
	for j=1:maxorder+1, i=1:maxorder+1
		A[j,i] = T(xs[j])^(i-1)
	end
	A \ T.(ys)
end

function graph_polynomial(gp::GraphProblem, ::Val{:finitefield}; usecuda=false, maxorder=graph_polynomial_maxorder(gp; usecuda=usecuda), max_iter=100)
    TI = Int32  # Int 32 is faster
    N = typemax(TI)
    YS = []
    local res
    for k = 1:max_iter
	    N = prevprime(N-TI(1))
        T = Mods.Mod{N,TI}
        rk = _polynomial_single(gp, T; usecuda=usecuda, maxorder=maxorder)
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

function optimize_code(code; method=:kahypar, sc_target=17, max_group_size=40, nrepeat=10, imbalances=0.0:0.001:0.8)
    size_dict = Dict([s=>2 for s in labels(code)])
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

function contractx(gp::GraphProblem, x::T; usecuda=false) where T
    xs = generate_tensors(_->x, gp)
    if usecuda
        xs = CuArray.(xs)
    end
    dynamic_einsum(gp.code, xs)
end

############### Problem specific implementations ################
### independent set ###
function Independence(g::SimpleGraph)
    Independence(EinCode(([(i,) for i in LightGraphs.vertices(g)]..., # labels for vertex tensors
                    [minmax(e.src,e.dst) for e in LightGraphs.edges(g)]...), ()))  # labels for edge tensors
end

function generate_tensors(fx, gp::Independence)
    ixs = getixs(flatten(gp.code))
    T = typeof(fx(ixs[1][1]))
    return map(ixs) do ix
        # if the tensor rank is 1, create a vertex tensor.
        # otherwise the tensor rank must be 2, create a bond tensor.
        length(ix)==1 ? misv(fx(ix[1])) : misb(T)
    end
end
misb(::Type{T}) where T = [one(T) one(T); one(T) zero(T)]
misv(val::T) where T = [one(T), val]

graph_polynomial_maxorder(gp::Independence; usecuda) = Int(sum(contractx(gp, TropicalF64(1.0); usecuda=usecuda)).n)

### coloring ###
Coloring(g::SimpleGraph) = Coloring(Independence(g).code)
function generate_tensors(fx, c::Coloring{K}) where K
    ixs = getixs(flatten(code))
    T = eltype(fx(ixs[1][1]))
    return map(ixs) do ix
        # if the tensor rank is 1, create a vertex tensor.
        # otherwise the tensor rank must be 2, create a bond tensor.
        length(ix)==1 ? coloringv(f(ix[1])) : coloringb(T, K)
    end
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
function Matching(g::SimpleGraph)
    Matching(EinCode(([(minmax(e.src,e.dst),) for e in LightGraphs.edges(g)]..., # labels for edge tensors
                    [([minmax(i,j) for j in neighbors(g, i)]...,) for i in LightGraphs.vertices(g)]...,), ()))        # labels for vertex tensors
end

function generate_tensors(fx, m::Matching)
    ixs = OMEinsum.getixs(flatten(m.code))
    T = typeof(fx(ixs[1][1]))
    n = length(unique(vcat(collect.(ixs)...)))  # number of vertices
    tensors = []
    for i=1:length(ixs)
        if i<=n
            @assert length(ixs[i]) == 1
            t = T[one(T), fx(ixs[i][1])] # fx is defined on edges.
        else
            t = match_tensor(T, length(ixs[i]))
        end
        push!(tensors, t)
    end
    return tensors
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

graph_polynomial_maxorder(m::Matching; usecuda) = Int(sum(contractx(m, TropicalF64(1.0); usecuda=usecuda)).n)

### maximal independent set ###
function MaximalIndependence(g::SimpleGraph)
    MaximalIndependence(EinCode(([(LightGraphs.neighbors(g, v)..., v) for v in LightGraphs.vertices(g)]...,), ()))
end

function generate_tensors(fx, mi::MaximalIndependence)
    ixs = OMEinsum.getixs(flatten(mi.code))
    T = eltype(fx(ixs[1][end]))
	return map(ixs) do ix
        neighbortensor(fx(ix[end]), length(ix))
    end
end
function neighbortensor(x::T, d::Int) where T
    t = zeros(T, fill(2, d)...)
    for i = 2:1<<(d-1)
        t[i] = one(T)
    end
    t[1<<(d-1)+1] = x
    return t
end

graph_polynomial_maxorder(mi::MaximalIndependence; usecuda) = Int(sum(contractx(mi, TropicalF64(1.0); usecuda=usecuda)).n)

for GP in [:Independence, :MaximalIndependence, :Matching, :Coloring]
    @eval optimize_code(gp::$GP; kwargs...) = $GP(optimize_code(gp.code; kwargs...))
end
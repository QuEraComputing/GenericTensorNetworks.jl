using Polynomials
using OMEinsum: NestedEinsum, getixs, getiy
using FFTW
using Graphs

export contractx, contractf, graph_polynomial, max_size, max_size_count

"""
    graph_polynomial(problem, method; usecuda=false, kwargs...)

Computing the graph polynomial for specific problem.

* `problem` can be one of the following instances,
    * `Independence` for the independence polynomial,
    * `MaximalIndependence` for the maximal independence polynomial,
    * `Matching` for the matching polynomial,

* `method` can be one of the following inputs,
    * `Val(:finitefield)`, compute exactly with the finite field method.
        It consumes additional kwargs [`max_iter`, `maxorder`], where `maxorder` is maximum order of polynomial
        and `max_iter` is the maximum number of primes numbers to use in the finitefield algebra.
        `max_iter` can be determined automatically in most cases.
    * `Val(:polynomial)`, compute directly with `Polynomial` number type,
    * `Val(:fft)`, compute with the fast fourier transformation approach, fast but needs to tune the hyperparameter `r`.
        It Consumes additional kwargs [`maxorder`, `r`]. The larger `r` is,
        the more accurate the factors of high order terms, and the less accurate the factors of low order terms.
    * `Val(:fitting)`, compute with the polynomial fitting approach, fast but inaccurate for large graphs.
"""
function graph_polynomial end

function graph_polynomial(gp::GraphProblem, ::Val{:fft}; usecuda=false, 
        maxorder=max_size(gp; usecuda=usecuda), r=1.0)
    ω = exp(-2im*π/(maxorder+1))
    xs = r .* collect(ω .^ (0:maxorder))
    ys = [Array(contractx(gp, x; usecuda=usecuda)) for x in xs]
    map(ci->Polynomial(ifft(getindex.(ys, Ref(ci))) ./ (r .^ (0:maxorder))), CartesianIndices(ys[1]))
end

function graph_polynomial(gp::GraphProblem, ::Val{:fitting}; usecuda=false,
        maxorder = max_size(gp; usecuda=usecuda))
    xs = (0:maxorder)
    ys = [Array(contractx(gp, x; usecuda=usecuda)) for x in xs]
    map(ci->fit(xs, getindex.(ys, Ref(ci))), CartesianIndices(ys[1]))
end

function graph_polynomial(gp::GraphProblem, ::Val{:polynomial}; usecuda=false)
    @assert !usecuda "Polynomial type can not be computed on GPU!"
    contractx(gp::GraphProblem, Polynomial([0, 1.0]))
end

function _polynomial_single(gp::GraphProblem, ::Type{T}; usecuda, maxorder) where T
	xs = 0:maxorder
    ys = [Array(contractx(gp, T(x); usecuda=usecuda)) for x in xs]  # download to CPU
    res = fill(T[], size(ys[1]))  # contraction result can be a tensor
    for ci in length(ys[1])
	    A = zeros(T, maxorder+1, maxorder+1)
        for j=1:maxorder+1, i=1:maxorder+1
            A[j,i] = T(xs[j])^(i-1)
        end
	    res[ci] = A \ T.(getindex.(ys, Ref(ci)))
    end
    return res
end

function graph_polynomial(gp::GraphProblem, ::Val{:finitefield}; usecuda=false,
        maxorder=max_size(gp; usecuda=usecuda), max_iter=100)
    return map(Polynomial, big_integer_solve(T->_polynomial_single(gp, T; usecuda=usecuda, maxorder=maxorder), Int32, max_iter))
end

function big_integer_solve(f, ::Type{TI}, max_iter::Int=100) where TI
    N = typemax(TI)
    local res, respre, YS
    for k = 1:max_iter
	    N = prevprime(N-TI(1))
        @debug "iteration $k, computing on GP($(N)) ..."
        T = Mods.Mod{N,TI}
        rk = f(T)
        if max_iter==1
            return map(x->BigInt.(Mods.value.(x)), rk)  # needs test
        end
        if k != 1
            push!.(YS, rk)
            res = map(x->improved_counting(x...), YS)
            all(respre .== res) && return res
            respre = res
        else  # k=1
            YS = reshape([Any[] for i=1:length(rk)], size(rk))
            push!.(YS, rk)
            respre = map(x->BigInt.(value.(x)), rk)
        end
    end
    @warn "result is potentially inconsistent."
    return res
end

function improved_counting(ys::AbstractArray...)
    map(yi->improved_counting(yi...), zip(ys...))
end
improved_counting(ys::Mod...) = Mods.CRT(ys...)

contractx(gp::GraphProblem, x; usecuda=false) = contractf(_->x, gp; usecuda=usecuda)
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

############### Problem specific implementations ################
### independent set ###
function generate_tensors(fx, gp::Independence)
    ixs = getixsv(gp.code)
    n = length(unique!(vcat(ixs...)))
    T = typeof(fx(ixs[1][1]))
    return map(enumerate(ixs)) do (i, ix)
        if i <= n
            misv(fx(ix[1]))
        else
            misb(T, length(ix)) # if n!=2, it corresponds to set packing problem.
        end
    end
end

function misb(::Type{T}, n::Integer=2) where T
    res = zeros(T, fill(2, n)...)
    res[1] = one(T)
    for i=1:n
        res[1+1<<(i-1)] = one(T)
    end
    return res
end
misv(val::T) where T = [one(T), val]

### coloring ###
function generate_tensors(fx, c::Coloring{K}) where K
    ixs = getixsv(c.code)
    T = eltype(fx(ixs[1][1]))
    return map(ixs) do ix
        # if the tensor rank is 1, create a vertex tensor.
        # otherwise the tensor rank must be 2, create a bond tensor.
        length(ix)==1 ? coloringv(fx(ix[1])) : coloringb(T, K)
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
function generate_tensors(fx, m::Matching)
    ixs = getixsv(m.code)
    T = typeof(fx(ixs[1][1]))
    n = length(unique!(vcat(ixs...)))  # number of vertices
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

### maximal independent set ###
function generate_tensors(fx, mi::MaximalIndependence)
    ixs = getixsv(mi.code)
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

### max cut/spin glass problem ###
function generate_tensors(fx, gp::MaxCut)
    ixs = getixsv(gp.code)
    return map(enumerate(ixs)) do (i, ix)
        maxcutb(fx(ix))
    end
end
function maxcutb(expJ::T) where T
    return T[one(T) expJ; expJ one(T)]
end

### paint shop ###
function generate_tensors(fx, c::PaintShop)
    ixs = getixsv(c.code)
    T = eltype(fx(ixs[1]))
    [paintshop_bond_tensor(fx(ixs[i]), c.isfirst[i], c.isfirst[i+1]) for i=1:length(ixs)]
end

function paintshop_bond_tensor(x::T, if1::Bool, if2::Bool) where T
    m = T[x one(T); one(T) x]
    m = if1 ? m : m[[2,1],:]
    m = if2 ? m : m[:,[2,1]]
    return m
end

for TP in [:MaximalIndependence, :Independence, :Matching, :MaxCut]
    @eval max_size(m::$TP; usecuda=false) = Int(sum(contractx(m, TropicalF64(1.0); usecuda=usecuda)).n)  # floating point number is faster (BLAS)
    @eval max_size_count(m::$TP; usecuda=false) = (r = sum(contractx(m, CountingTropical{Float64,Float64}(1.0, 1.0); usecuda=usecuda)); (Int(r.n), Int(r.c)))
end

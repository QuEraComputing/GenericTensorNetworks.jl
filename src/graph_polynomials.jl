using Polynomials
using OMEinsum: NestedEinsum, getixs, getiy
using FFTW
using Graphs

"""
    graph_polynomial(problem, method; usecuda=false, kwargs...)

Computing the graph polynomial for specific problem.

* `problem` can be one of the following instances,
    * `IndependentSet` for the independence polynomial,
    * `MaximalIS` for the maximal independence polynomial,
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
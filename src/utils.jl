# Return a vector of unique labels in an Einsum token.
function labels(code::AbstractEinsum)
    res = []
    for ix in getixsv(code)
        for l in ix
            if l ∉ res
                push!(res, l)
            end
        end
    end
    return res
end

# a unified interface to optimize the contraction code
_optimize_code(code, size_dict, optimizer::Nothing, simplifier) = code
_optimize_code(code, size_dict, optimizer, simplifier) = optimize_code(code, size_dict, optimizer, simplifier)

# upload tensors to GPU
function togpu end
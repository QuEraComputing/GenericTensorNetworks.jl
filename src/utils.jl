export is_independent_set

function is_independent_set(g::SimpleGraph, config)
    for e in edges(g)
        if config[e.src] == config[e.dst] == 1
            return false
        end
    end
    return true
end
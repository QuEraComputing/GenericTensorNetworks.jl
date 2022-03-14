using GraphTensorNetworks, Graphs

function k23()
    g = SimpleGraph(5)
    for (i, j) in [(1,5), (1,4), (1,2), (3,2), (3,4), (3, 5)]
        add_edge!(g, i, j)
    end
    return g
end

function mapped_k23()
    g = SimpleGraph(9)
    for (i, j) in [(1,5), (1,4), (1,7), (2,8), (2,9),
        (3, 5), (3, 4), (3, 6), (4,8), (4, 9), (6, 8),
        (7,9),
        ]
        add_edge!(g, i, j)
    end
    return g
end

function compute_spectrums(problem, nlevel, weights_list)
    res = zeros(nlevel, length(weights_list))
    for i=1:length(weights_list)
        weights = copy(problem.weights)
        weights[1:length(weights_list[i])] .+= weights_list[i]
        prob = IndependentSet(problem.code, length(weights), weights)
        res[:,i] .= solve(prob, SizeMax(nlevel))[].orders
    end
    return res
end

source = Independence(k23(), weights=zeros(5))
mapped_weight0 = [1.0, 0, 1, 0, 0, 2, 2, 1, 1]
mapped = Independence(mapped_k23(), weights=mapped_weight0)
nlevel = Int(solve(source, CountingAll())[])

weights_list = [[0.01+0.01*sin(2π*t/(1<<(k-1))) for k=1:5] for t=0.0:0.1:16.0]
spectrum1 = compute_spectrums(source, nlevel, weights_list)
spectrum2 = compute_spectrums(mapped, nlevel, weights_list)

using PyPlot
PyPlot.subplot(211)
PyPlot.plot(spectrum1')
PyPlot.subplot(212)
PyPlot.plot(spectrum2')
PyPlot.xlabel("weights")
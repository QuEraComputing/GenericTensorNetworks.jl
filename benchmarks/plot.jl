using Plots, DelimitedFiles
using Plots: cm

function plot_benchmark()
    # nw = [(10, 3), (20, 4), (30, 6), (40, 7), (50, 8), (60, 10), (70, 11), (80, 11), (90, 15), (100, 15),
    #    (110, 15), (120, 18), (130, 17), (140, 16), (150, 21), (160, 20), (170, 22), (180, 24), (190, 26), (200, 25)]
    nw = [(10, 3), (20, 4), (30, 5), (40, 6), (50, 8), (60, 9), (70, 8), (80, 11), (90, 13), (100, 13),
            (110, 15), (120, 16), (130, 14), (140, 18), (150, 18), (160, 22), (170, 19), (180, 25), (190, 24), (200, 26),]
    ns = getindex.(nw, 1)
    plt = Plots.plot([], []; yscale=:log10, label="", xtickfontsize=12,ytickfontsize=12,xlabel="Graph size",xguidefontsize=14,ylabel="time/s",yguidefontsize=14,legendfontsize=8,legend=:topleft, size=(600, 400), right_margin=1.5cm,fg_legend = :transparent)
    plt = Plots.plot!(twinx(), ns, getindex.(nw, 2), label="tree width",ytickfontsize=12, xticks=:none,yguidefontsize=14,legend=:bottomright, color=:black, ls=:dash, lw=2, ylabel="tree width")
    for (prefix, l) in [("totalsize", "total size"), ("maxsize", "max size"), ("idp_polynomial", "IDP (polynomial)"), ("idp_fft", "IDP (FFT)"), ("idp_finitefield", "IDP (finite field)"),
        ("config_single", "single configuration"), ("config_single_bounded", "single configuration (bounding)"), ("config_all", "all configurations"), ("config_all_bounded", "all configurations (bounding)")]
        datafile = joinpath(pwd(), "benchmarks", "data", prefix * "-r3-CPU.dat")
        y = readdlm(datafile) ./ 1e9
        plot!(plt, ns[1:length(y)], y; label=l, lw=2)
    end
    savefig("paper/benchmark.pdf")
    plt
end

plot_benchmark()
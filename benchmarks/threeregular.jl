using NoteOnTropicalMIS, DelimitedFiles
using LightGraphs
using BenchmarkTools

function case_r3(n, k=3; sc_target, seed=2)
    # generate a random regular graph of size 100, degree 3
    graph = (Random.seed!(seed); LightGraphs.random_regular_graph(n, k))
    @assert length(connected_components(graph)) == 1  # connected graph
    # optimize the contraction order using KaHyPar + Greedy
    optcode = idp_code(graph; method=:kahypar, sc_target=sc_target, max_group_size=40, imbalances=0:0.001:1)
    return optcode
end

function case_dc(L::Int, ρ; sc_target, seed=2)
    # generate a random regular graph of size 100, degree 3
    Random.seed!(seed)
    graph = diagonal_coupled_graph(rand(L, L) .< ρ)
    # optimize the contraction order using KaHyPar + Greedy, target space complexity is 2^20
    optcode = idp_code(graph; method=:kahypar, sc_target=sc_target, max_group_size=40)
    return optcode
end

function case_sq(L::Int, ρ; sc_target, seed=2)
    # generate a random regular graph of size 100, degree 3
    Random.seed!(seed)
    graph = square_lattice_graph(rand(L, L) .< ρ)
    # optimize the contraction order using KaHyPar + Greedy, target space complexity is 2^20
    optcode = idp_code(graph; method=:kahypar, sc_target=sc_target, max_group_size=40)
    return optcode
end

# setup global arguments
const GRAPH = length(ARGS) >= 1 ? ARGS[1] : "r3"
const TASK = length(ARGS) >= 2 ? ARGS[2] : "maxsize"
const DEVICE = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : -1

if DEVICE >= 0
    using CUDA
    CUDA.device!(DEVICE)
end

function run_benchmarks(cases; output_file)
    suite = BenchmarkGroup()
    for (case, f) in cases
        suite[case] = @benchmarkable $f()
    end

    tune!(suite)
    res = run(suite)

    times = zeros(length(cases))
    for (k, (case, f)) in enumerate(cases)
        times[k] = minimum(res[case].times)
    end

    println("Writing benchmark results to file: $output_file")
    mkpath(dirname(output_file))
    writedlm(output_file, times)
end

function runcase(;
        case_set=:r3,
        task = :maxsize,
        usecuda = false,
        ntruncate = 0  # truncate benchmark cases
    )
    cases = if case_set == :r3
        [case_r3(n, 3; seed=2, sc_target=s) for (n, s) in [
            (10, 3), (20, 4), (30, 5), (40, 6), (50, 8), (60, 9), (70, 8), (80, 11), (90, 13), (100, 13),
            (110, 15), (120, 16), (130, 14), (140, 18), (150, 18), (160, 22), (170, 19), (180, 25), (190, 24), (200, 26),
        ][1:end-ntruncate]]
    else
        [case_dc(L, 0.8; seed=2, sc_target=s) for (L, s) in [
            (4, 5), (6, 7), (8, 9), (10, 8), (12, 12), (14, 13), (16, 17), (18, 18), (20, 18), (22, 23), (24, 23),
        ][1:end-ntruncate]]
    end

    run_benchmarks([("n$(10*i)", ()->run_task(case, task)) for (i, case) in enumerate(cases)],
                output_file=NoteOnTropicalMIS.project_relative_path("benchmarks", "data", "$(task)-$(case_set)-$(usecuda ? "GPU" : "CPU").dat"))
end

const truncatedict = Dict(
    "r3"=>Dict([string(task)=>ntruncate for (task, ntruncate) in [
        (:totalsize, 0), (:maxsize, 0), (:counting, 0),
        (:idp_polynomial, 3), (:idp_fft, 0), (:idp_finitefield, 3),
        (:config_single, 0), (:config_all, 6), (:config_single_bounded, 0), (:config_all_bounded, 3)
        ]]),
    "dc"=>Dict([string(task)=>ntruncate for (task, ntruncate) in [
        (:totalsize, 0), (:maxsize, 0), (:counting, 0),
        (:idp_polynomial, 0), (:idp_fft, 0), (:idp_finitefield, 0),
        (:config_single, 0), (:config_all, 5), (:config_single_bounded, 0), (:config_all_bounded, 0)
    ]]))
runcase(case_set=Symbol(GRAPH), task=Symbol(TASK), usecuda=DEVICE>=0, ntruncate=truncatedict[GRAPH][TASK])

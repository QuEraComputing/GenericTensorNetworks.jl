using Pkg
using GraphTensorNetworks
using Documenter
using Literate

for each in readdir(pkgdir(GraphTensorNetworks, "examples"))
    project_dir = pkgdir(GraphTensorNetworks, "examples", each)
    isdir(project_dir) || continue
    @info "building" project_dir
    input_file = pkgdir(GraphTensorNetworks, "examples", each, "main.jl")
    output_dir = pkgdir(GraphTensorNetworks, "docs", "src", "tutorials")
    @info "executing" input_file
    Literate.markdown(input_file, output_dir; name=each)
end

DocMeta.setdocmeta!(GraphTensorNetworks, :DocTestSetup, :(using GraphTensorNetworks); recursive=true)

makedocs(;
    modules=[GraphTensorNetworks],
    authors="Jinguo Liu",
    repo="https://github.com/Happy-Diode/GraphTensorNetworks.jl/blob/{commit}{path}#{line}",
    sitename="GraphTensorNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Happy-Diode.github.io/GraphTensorNetworks.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Independent Set Problem" => "tutorials/IndependentSet.md",
            "Max-Cut Problem" => "tutorials/MaxCut.md",
        ],
        "References" => "ref.md",
    ],
)

deploydocs(;
    repo="github.com/Happy-Diode/GraphTensorNetworks.jl",
)

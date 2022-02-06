using Pkg
using GraphTensorNetworks
using GraphTensorNetworks: TropicalNumbers, Polynomials, Mods
using Documenter
using DocThemeIndigo
using Literate

for each in readdir(pkgdir(GraphTensorNetworks, "examples"))
    project_dir = pkgdir(GraphTensorNetworks, "examples", each)
    isdir(project_dir) || continue
    @info "building" project_dir
    input_file = pkgdir(GraphTensorNetworks, "examples", each, "main.jl")
    output_dir = pkgdir(GraphTensorNetworks, "docs", "src", "tutorials")
    @info "executing" input_file
    Literate.markdown(input_file, output_dir; name=each, execute=false)
end

indigo = DocThemeIndigo.install(GraphTensorNetworks)
DocMeta.setdocmeta!(GraphTensorNetworks, :DocTestSetup, :(using GraphTensorNetworks); recursive=true)

makedocs(;
    modules=[GraphTensorNetworks, TropicalNumbers, Polynomials, Mods],
    authors="Jinguo Liu",
    repo="https://github.com/Happy-Diode/GraphTensorNetworks.jl/blob/{commit}{path}#{line}",
    sitename="GraphTensorNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Happy-Diode.github.io/GraphTensorNetworks.jl",
        assets=String[indigo],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Independent Set Problem" => "tutorials/IndependentSet.md",
            "Max-Cut Problem" => "tutorials/MaxCut.md",
            "Other Problems" => "tutorials/Others.md",
        ],
        "Method Selection Guide" => "methodselection.md",
        "References" => "ref.md",
    ],
    doctest=false,
)

deploydocs(;
    repo="github.com/Happy-Diode/GraphTensorNetworks.jl",
)

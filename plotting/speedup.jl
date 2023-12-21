

# call make_plot("syrk", "L") if measurements already collected
# make_plot("syrk", "L", collect=true) otherwise, will collect measurements for all languages
# make_plot("syrk, "L", collect=true, languages=[:python, :julia, :C]) languages kwarg to specify languages, only collect measurements for these languages
# make_plot("syrk, "L", collect=true, languages=[:C], version="naive") to recollect a specific version

using CSV, DataFrames
using Plots


include("../timing/collect_measurements.jl")

# assumes there exists a C version called "naive", and a cuda C version called "naive_gpu"


function make_plots(preset)
    bms = ["covariance", "doitgen", "floyd-warshall", "gemm", "lu", "syrk", "trisolv", "jacobi_2d"]
    for bm in bms
        try
            make_plot(bm, preset)
        catch e end
    end
end




plot_versions(plot, df, version_apdx, color, label) = bar!(plot, 
    df.version .* version_apdx, df.speedup, 
    yerror=(df.speedup-df.speedup_lb, df.speedup_ub-df.speedup), 
    color=color, 
    label=label, 
    bar_width=0.7
)

function add_speedup!(df, base)
    df.speedup = base ./ df.median
    df.speedup_lb = base ./ df.median_ub
    df.speedup_ub = base ./ df.median_lb
end

function plot_arch(bm, preset, c_versions, julia_versions, python_versions, arch)
    python_blue = RGBA(48/255, 101/255, 166/255)
    julia_green = RGBA(23/255, 145/255, 62/255)

    cd(@__DIR__)
    
    p = bar(
        title="Speedup $bm, preset '$preset', $arch", 
        ylabel="Speedup", 
        xlabel=" ", # it will give more space at the bottom this way
        xrotation=30,
        ylims=(0, 1.1*max(
            maximum(c_versions.speedup_ub, init=0.0),
            maximum(julia_versions.speedup_ub, init=0.0),
            maximum(python_versions.speedup_ub, init=0.0),
        ))
    )
    plot_versions(p, c_versions, " (C)", :gray, "C")
    plot_versions(p, julia_versions, " (J)", julia_green, "julia")
    plot_versions(p, python_versions, " (P)", python_blue, "python")
    display(p)
    savefig(p, "plots/speedup_$(bm)_$(preset)_$(arch).png")
end

function make_plot(bm::String, preset::String; collect=false, version=nothing, languages=[:julia, :C, :python])
    cd(@__DIR__)
    cd("..")
    if collect
        collect_measurements([bm], languages, [preset], version)
    end
    df = CSV.read("results.csv", DataFrame)
    grouped = groupby(df, [:benchmark, :language, :preset, :gpu])

    short_name = NPBenchManager.get_short_name(bm)

    julia_versions      = haskey(grouped, (short_name, "julia", preset, false)) ? grouped[(short_name, "julia", preset, false)] : empty_df()
    c_versions          = haskey(grouped, (short_name, "C",     preset, false)) ? grouped[(short_name, "C",     preset, false)] : empty_df()
    python_versions     = haskey(grouped, (short_name, "python",preset, false)) ? grouped[(short_name, "python",preset, false)] : empty_df()
    
    
    versions = [julia_versions, c_versions, python_versions]
    i = findfirst(v->v=="naive", c_versions.version)
    if isnothing(i)
        error("need to define a c version called 'naive'!")
    end
    c_naive = c_versions[[i], :]
    ref_time = c_naive[1, :median]

    for v in versions
        add_speedup!(v, ref_time)
    end

    plot_arch(bm, preset, c_versions, julia_versions, python_versions, "CPU")



    julia_gpu_versions  = haskey(grouped, (short_name, "julia", preset, true)) ? grouped[(short_name, "julia", preset, true)] : empty_df()
    c_gpu_versions      = haskey(grouped, (short_name, "C",     preset, true)) ? grouped[(short_name, "C",     preset, true)] : empty_df()
    python_gpu_versions = haskey(grouped, (short_name, "python",preset, true)) ? grouped[(short_name, "python",preset, true)] : empty_df()

    versions = [julia_gpu_versions, c_gpu_versions, python_gpu_versions]
    i = findfirst(v->v=="naive_gpu", c_gpu_versions.version)
    if isnothing(i)
        error("need to define a cuda c version called 'naive_gpu'!")
    end
    c_naive_gpu = c_gpu_versions[[i], :]
    ref_time = c_naive_gpu[1, :median]
    
    for v in versions
        add_speedup!(v, ref_time)
    end

    plot_arch(bm, preset, c_gpu_versions, julia_gpu_versions, python_gpu_versions, "GPU")
    ;
end

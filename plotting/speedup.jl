
try
    using CSV, DataFrames
catch e
    import Pkg
    Pkg.add("CSV")
    Pkg.add("DataFrames")
    using CSV, DataFrames
end
try
    using Plots
catch e
    import Pkg
    Pkg.add("Plots")
    using Plots
end

# assumes there exists a C version called "naive", and a cuda C version called "naive_gpu"


plot_versions(plot, df, version_apdx, color; label=df[1, :language]) = bar!(plot, 
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
        xrotation=45,
        ylims=(0, 1.1*max(
            maximum(c_versions.speedup_ub),
            maximum(julia_versions.speedup_ub),
            maximum(python_versions.speedup_ub),
        ))
    )
    plot_versions(p, c_versions, " (C)", :gray)
    plot_versions(p, julia_versions, " (J)", julia_green)
    plot_versions(p, python_versions, " (P)", python_blue)
    display(p)
    savefig(p, "plots/speedup_$(bm)_$(preset)_$(arch).pdf")
end

function make_plot(bm::String, preset::String; collect=false)
    cd(@__DIR__)
    cd("..")
    if collect
        run(`julia run_benchmarks.jl -b$(bm) -p$(preset) -lcjp`)
    end
    df = CSV.read("results.csv", DataFrame)
    grouped = groupby(df, [:benchmark, :language, :preset, :gpu])

    julia_versions      = grouped[(bm, "julia", preset, false)]
    c_versions          = grouped[(bm, "C",     preset, false)]
    python_versions     = grouped[(bm, "python",preset, false)]
    
    
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



    julia_gpu_versions  = grouped[(bm, "julia", preset, true)]
    c_gpu_versions      = grouped[(bm, "C",     preset, true)]
    python_gpu_versions = grouped[(bm, "python",preset, true)]

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

end


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

# assumes there exists a C version called "naive"


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
    julia_gpu_versions  = grouped[(bm, "julia", preset, true)]
    c_gpu_versions      = grouped[(bm, "C",     preset, true)]
    python_gpu_versions = grouped[(bm, "python",preset, true)]

    versions = [julia_versions, c_versions, python_versions, julia_gpu_versions, c_gpu_versions, python_gpu_versions]
    c_naive = c_versions[[findfirst(v->v=="naive", c_versions.version)], :]
    ref_time = c_naive[1, :median]

    for v in versions
        add_speedup!(v, ref_time)
    end
    
    c_naive = c_versions[[findfirst(v->v=="naive", c_versions.version)], :]

    python_blue = RGBA(48/255, 101/255, 166/255)
    julia_green = RGBA(23/255, 145/255, 62/255)

    cd(@__DIR__)
    
    p = bar(
        title="Speedup $bm, preset '$preset', CPU", 
        ylabel="Speedup", 
        xlabel="version", 
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
    savefig(p, "plots/speedup_$(bm)_$(preset)_cpu.pdf")

    p = bar(
        title="Speedup $bm, preset '$preset', GPU", 
        ylabel="Speedup", 
        xlabel="version", 
        xrotation=45,
        ylims=(0, 1.1*max(1, 
            maximum(c_gpu_versions.speedup_ub),
            maximum(julia_gpu_versions.speedup_ub),
            maximum(python_gpu_versions.speedup_ub),
        ))
    )
    plot_versions(p, c_naive, " (C CPU)", :white, label="C CPU")
    plot_versions(p, c_gpu_versions, " (C)", :gray)
    plot_versions(p, julia_gpu_versions, " (J)", julia_green)
    plot_versions(p, python_gpu_versions, " (P)", python_blue)
    display(p)
    savefig(p, "plots/speedup_$(bm)_$(preset)_gpu.pdf");
end

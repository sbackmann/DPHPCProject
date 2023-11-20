
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


plot_versions(plot, df, version_apdx, color) = bar!(plot, 
    df.version .* version_apdx, df.speedup, 
    yerror=(df.speedup-df.speedup_lb, df.speedup_ub-df.speedup), 
    color=color, 
    label=df[1, :language], 
    bar_width=0.7
)

function add_speedup(df, base)
    df.speedup = base ./ df.median
    df.speedup_lb = base ./ df.median_ub
    df.speedup_ub = base ./ df.median_lb
end

function make_plot(bm::String, preset::String; collect=false)
    if collect
        cd(@__DIR__)
        cd("..")
        run(`julia run_benchmarks.jl -b$(bm) -p$(preset) -lcjp`)
    end
    df = CSV.read("results.csv", DataFrame)
    grouped = groupby(df, [:benchmark, :language, :preset])
    julia_versions = grouped[(bm, "julia",preset)]
    c_versions = grouped[(bm, "C",preset)]
    python_versions = grouped[(bm, "python",preset)]

    ref_time = c_versions[findfirst(v->v=="naive", c_versions.version), :median]
    add_speedup(c_versions, ref_time)
    add_speedup(julia_versions, ref_time)
    add_speedup(python_versions, ref_time)
    
    p = bar(title="Speedup $bm, preset '$preset'", ylabel="Speedup", xlabel="version", xrotation=45)
    plot_versions(p, c_versions, " (C)", :gray)
    plot_versions(p, julia_versions, " (J)", :green)
    plot_versions(p, python_versions, " (P)", :blue)
    display(p)
end

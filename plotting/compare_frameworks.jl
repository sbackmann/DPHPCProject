using CSV, DataFrames, StatsPlots
include("../timing/NPBenchManager.jl")

benchmarks = ["covariance", "doitgen", "floyd_warshall", "gemm", "jacobi_2d", "lu", "syrk", "trisolv"]
bms_short  = NPBenchManager.get_short_name.(benchmarks)
frameworks = ["numpy", "dace", "numba", "pythran", "cupy"]

function get_framework(s::String)
    for fw in frameworks 
        if occursin(fw, s)
            return fw
        end
    end
    error("hi, uuhm, you know, the framework, i dont...")
end

function split_python_versions(df)
    gdf = groupby(df, [:version])
    tf = transform(gdf, :version => (x -> get_framework.(x)) => :framework)
    gfw = groupby(tf, [:benchmark, :framework])
    vs = []
    infinity = 1e9
    for fw in frameworks
        new_df = df[2:1, :]
        for bm in bms_short
            if haskey(gfw, (bm, fw))
                new_df = vcat(new_df, gfw[(bm, fw)][:, names(df)])
            else
                new_df = vcat(new_df, 
                    DataFrame(
                        benchmark=bm, 
                        language="python", 
                        version="dummy", 
                        preset="paper", 
                        gpu=df[1, :gpu], 
                        median=infinity, 
                        median_lb=infinity, 
                        median_ub=infinity, 
                        nr_runs=0
                    )
                )
            end
        end
        push!(vs, new_df)
    end
    return vs
end

function best_version(df)
    df[argmin(df.median), :]
end

function get_best_versions(df)
    gdf = groupby(df, [:benchmark])
    best = df[2:1, :]
    for g in gdf
        push!(best, best_version(g))
    end
    best
end

function fill_library_gaps!(df, gpu)
    infinity=1e9
    for bm in bms_short
        if isnothing(findfirst(x->x==bm, df.benchmark))
            append!(df, DataFrame(
                benchmark=bm, 
                language="julia", 
                version="dummy", 
                preset="paper", 
                gpu=gpu, 
                median=infinity, 
                median_lb=infinity, 
                median_ub=infinity, 
                nr_runs=0
            ))
        end
    end
end

function get_relevant_versions(gpu::Bool)
    cd(@__DIR__)
    cd("..")
    df = CSV.read("results.csv", DataFrame)
    gdf = groupby(df, [:language, :preset, :gpu])
    c, julia, python = (l->gdf[(l, "paper", gpu)]).(("C", "julia", "python")) 
    c, julia, python = (df->df[(bm->bm ∈ bms_short).(df.benchmark), :]).((c, julia, python)) 

    is_library = julia[:, :version] .== "library" .|| julia[:, :version] .== "library_gpu"
    library = julia[is_library,   :]
    fill_library_gaps!(library, gpu)
    julia   = julia[.!is_library, :]
    numpy, dace, numba, pythran, cupy = split_python_versions(python)
    versions = [c, julia, library, numpy, dace, numba, pythran, cupy]
    best_versions = get_best_versions.(versions)
end

function group_benchmarks(versions)
    grouped = []
    for bm in bms_short
        g = versions[1][2:1, :]
        for v in versions
            gbm = groupby(v, [:benchmark])
            g = vcat(g, gbm[(bm,)])
        end
        push!(grouped, g)
    end
    grouped
end

function add_performance!(df)
    df.P    = 1 ./ df.median
    max_P = maximum(df.P)
    df.P    = df.P ./ max_P
    df.P_ub = (1 ./ df.median_lb) ./ max_P
    df.P_lb = (1 ./ df.median_ub) ./ max_P
end

function make_plot(gpu::Bool, frameworks_to_compare; title="", colors=nothing)
    versions = get_relevant_versions(gpu)
    g = group_benchmarks(versions)
    all_fws = ["C", "julia", "library", frameworks...]
    indeces = [findfirst(fw->fw==x, all_fws) for x in frameworks_to_compare]
    g = [v[indeces, :] for v in g]
    add_performance!.(g)
    M = Matrix{Float64}(undef, 8, length(frameworks_to_compare))
    for (i,x) in enumerate(g)
        M[i, :] .= x.P
    end
    display(M)
    yticks = collect(0:0.5:1.1)
    
    groupedbar(
        repeat(benchmarks, outer=length(frameworks_to_compare)), M,
        bar_position = :dodge,
        # size=(1000, 200),
        yticks=(yticks, string.(round.(100 .* yticks) .|> Int).*"%"),
        group = repeat(all_fws[indeces], inner=8),
        bar_width=0.5,
        title=title,
        legend_position=:outerleft,
        # ylabel="Performance",
        ylabelfontsize=9,
        color=repeat(colors, inner=8)
    )
end




function make_plot()
    juliac  = RGB{Float64}(0.1725,0.6275,0.1725)
    pythonc = RGB{Float64}(1.0,0.498,0.0549)#RGBA(217/255, 95/255, 2/255)
    libc    = RGB{Float64}(0.1216,0.4667,0.7059)
    p0 = make_plot(false, ["julia", "library", "numpy"], title="Julia vs Numpy"  , colors=[juliac, libc, pythonc])
    p1 = make_plot(false, ["julia", "library", "numba"], title="Julia vs Numba"  , colors=[juliac, libc, pythonc])
    p2 = make_plot(false, ["julia", "library", "dace"],  title="Julia vs DaCe"   , colors=[juliac, libc, pythonc])
    p3 = make_plot(false, ["julia", "library", "pythran"],          title="Julia vs Pythran", colors=[juliac, libc, pythonc])
    P = plot(plot(grid=false, showaxis=false),
        p0, p1, p2, p3,
        plot_title="Performance Comparison, Julia vs Python, CPU\n(higher is better)",
        layout=@layout([a{0.01h};°; °; °; °]),
        size=(1000, 600),
        bottom_margin=15*Plots.px,
        top_margin=4*Plots.px,
        right_margin=5*Plots.px,
        left_margin=10*Plots.px,
    )
    cd(@__DIR__)
    savefig(P, "plots/julia_vs_python.pdf")
    P
end

make_plot()
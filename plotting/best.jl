using CSV, DataFrames, Plots
include("../timing/NPBenchManager.jl")

green = RGBA(23/255, 145/255, 62/255)
red   = RGBA(232/255, 16/255, 38/255)
blue  = RGBA(48/255, 101/255, 166/255)
gray  = :gray

function get_library_version(df)
    is_library = df[:, :version] .== "library" .|| df[:, :version] .== "library_gpu"
    return df[is_library, :], df[.!is_library, :]
end

function get_best_version(df)
    @assert size(df, 1) > 0
    sorted = sort(df, :median)
    return sorted[[1], :]
end

function get_best_versions(bm::String, preset::String)
    cd(@__DIR__)
    cd("..")
    df = CSV.read("results.csv", DataFrame)
    grouped = groupby(df, [:benchmark, :language, :preset, :gpu])

    name = NPBenchManager.get_short_name(bm)

    c_cpu      = grouped[(name, "C",      preset, false)]
    julia_cpu  = grouped[(name, "julia",  preset, false)]
    python_cpu = grouped[(name, "python", preset, false)]

    c_gpu      = grouped[(name, "C",      preset, true)]
    julia_gpu  = grouped[(name, "julia",  preset, true)]
    python_gpu = grouped[(name, "python", preset, true)]

    library_cpu, julia_cpu = get_library_version(julia_cpu)
    library_gpu, julia_gpu = get_library_version(julia_gpu)

    c_cpu, julia_cpu, python_cpu = get_best_version.((c_cpu, julia_cpu, python_cpu))
    c_gpu, julia_gpu, python_gpu = get_best_version.((c_gpu, julia_gpu, python_gpu))

    cpu = c_cpu, python_cpu, julia_cpu, library_cpu
    gpu = c_gpu, python_gpu, julia_gpu, library_gpu

    return cpu, gpu
end

function add_performance!(df)
    df.performance    = 1 ./ df.median
    df.performance_ub = 1 ./ df.median_lb
    df.performance_lb = 1 ./ df.median_ub
end

function make_sub_plot(c, python, julia, library, bm)
    add_performance!.((c, python, julia, library))
    max_performance =    max(c[1, :performance],    python[:, :performance]...,    julia[1, :performance],    library[:, :performance]...)
    max_performance_ub = max(c[1, :performance_ub], python[:, :performance_ub]..., julia[1, :performance_ub], library[:, :performance_ub]...)
    yticks = collect(0:0.25:1.1)
    P = bar(

        yticks=(yticks, string.(round.(100 .* yticks) .|> Int).*"%"),
        # ylabel="Performance",
        ylims=(0,max(1.1, 1.05*max_performance_ub/max_performance)),
        xrotation=20,
        legend=false
    )

    function get_yerror(df)
        (df.performance    - df.performance_lb, 
         df.performance_ub - df.performance) ./ max_performance
    end

    bar!(P, 
        c.version .* " (C)", c.performance ./ max_performance, 
        yerror=get_yerror(c),
        color=gray,
    )
    
    bar!(P, 
        julia.version .* " (J)", julia.performance ./ max_performance, 
        yerror=get_yerror(julia),
        color=green,
    )

    bar!(P, 
        library.version .* " (J)", library.performance ./ max_performance, 
        yerror=get_yerror(library),
        color=red,
    )

    bar!(P, 
        python.version .* " (P)", python.performance ./ max_performance, 
        yerror=get_yerror(python),
        color=blue,
    )


    return P

end

function combine_plots(plots, arch, bms, preset)
    legend = bar(grid=false, showaxis=false, 
        legend_columns=4, legend_position=:topleft,
        legendfontsize = 11,
    )
    for (l, c) in [" C" => gray, " NPBench    " => blue, " Julia" => green, " Julia library" => red]
        bar!(legend, [], [], label=l, color=c)
    end
    the_plot = plot(
        legend, plots..., layout=@layout([a{0.01h} _ _ _; ° ° ° °; ° ° ° °]),
        # cpu_plots..., layout=(2, 4),
        plot_title="Performance Comparison of Best Versions $arch\n(higher is better)",
        title=["" bms],
        size=(1200, 650),
        bottom_margin=30*Plots.px,
        top_margin=23*Plots.px,
        right_margin=10*Plots.px,
        left_margin=40*Plots.px,
    )
    return the_plot
end

function make_single_plot(bm, preset)
    cpu_vs, gpu_vs = get_best_versions(bm, preset)
    display(make_sub_plot(cpu_vs..., bm))
    display(make_sub_plot(gpu_vs..., bm))
end

function make_plot(preset)
    cpu_plots = []
    gpu_plots = []
    bms = ["covariance" "doitgen" "floyd_warshall" "gemm" "jacobi_2d" "lu" "syrk" "trisolv"]
    for bm in bms
        cpu_vs, gpu_vs = get_best_versions(bm, preset)
        push!(cpu_plots, make_sub_plot(cpu_vs..., bm))
        push!(gpu_plots, make_sub_plot(gpu_vs..., bm))
    end

    cpu_plot = combine_plots(cpu_plots, "CPU", bms, preset)
    gpu_plot = combine_plots(gpu_plots, "GPU", bms, preset)
    
    display(cpu_plot)
    display(gpu_plot)

    cd(@__DIR__)
    savefig(cpu_plot, "plots/perfcomp_cpu_$(preset).pdf")
    savefig(gpu_plot, "plots/perfcomp_gpu_$(preset).pdf")
end
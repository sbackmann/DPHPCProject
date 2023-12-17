using CSV, DataFrames, Plots

include("../timing/collect_measurements.jl")

# percentage speedup: (c - julia) / c
# 50%   speedup: julia twice as fast as c
# 10%   speedup: julia = 0.9*c runtime  10% faster
# -10%  speedup: julia = 1.1*c runtime, 10% slower
# -100% speedup: julia twice as slow as c


function make_plots(preset)
    make_plot(preset, "naive", false)
    make_plot(preset, "naive_gpu", true)
    make_plot(preset, false)
    make_plot(preset, true)
end


function collect_version(presets::Vector{String}, version::String)
    bms = get_benchmarks()
    for bm in bms
        collect_measurements([bm], [:julia, :C], presets, version)
    end
end

collect_naive(presets::Vector) = collect_version(presets, "naive")
collect_naive(preset::String) = collect_version([preset], "naive")

collect_naive_gpu(presets::Vector) = collect_version(presets, "naive_gpu")
collect_naive_gpu(preset::String) = collect_version([preset], "naive_gpu")

# you business to collect the other versions...



ticks(min, max, num) = round.([min + f*(max - min) for f=0.0:1/(num-1+1e-4):1.0+0.5/(num-1+1e-4)], digits=1)

function plot_julia_vs_c(bms, c_times, julia_times, name, preset)
    percentage_speedup =  100 .* (c_times .- julia_times) ./ c_times

    s = sortperm(percentage_speedup, rev=true)
    percentage_speedup = percentage_speedup[s]
    bms = bms[s]

    mn, mx = (x->(minimum(x),maximum(x)))(percentage_speedup)
    L = 600 # limit for plot
    mn = clamp(mn, -L, 0)
    mx = clamp(mx, 0, L)
    

    faster = percentage_speedup .>= 0
    slower = percentage_speedup .<  0

    total_ticks = 7
    faster_ticks = (mx / (mx - mn) * (total_ticks-1) |> round |> Int) + 1
    slower_ticks = total_ticks - faster_ticks + 1

    yticks = [ticks(0, mn, slower_ticks); ticks(0, mx, faster_ticks)]

    alpha = 0.9
    green = RGBA(23/255, 145/255, 62/255, alpha)
    red   = RGBA(232/255, 16/255, 38/255, alpha)
    colors = [fill(green, sum(faster)); fill(red, sum(slower))]

    # plot faster ones in green
    P = bar(
        bms, percentage_speedup, 
        yticks=(yticks, string.(yticks).*"%"),

        title="Speedup: Julia over C, $name versions, preset $preset",
        titlefontsize=12,
        ylabel="Speedup vs C",
        color=colors,
        legend=false,
        ylims=[1.1*clamp(mn, -L, -9), 1.1*clamp(mx, 9, L)],
    )

    display(P)
    cd(@__DIR__)
    savefig(P, "plots/julia_vs_c_$(name)_$(preset).pdf")
end


function make_plot(preset::String, version::String, gpu::Bool)
    cd(@__DIR__)
    cd("..")
    df = CSV.read("results.csv", DataFrame)
    grouped = groupby(df, [:language, :preset, :gpu])

    c_versions = grouped[("C", preset, gpu)]
    julia_versions = grouped[("julia", preset, gpu)]

    c_versions = c_versions[c_versions[:, "version"] .== version, :]
    julia_versions = julia_versions[julia_versions[:, "version"] .== version, :]

    c_versions = c_versions[:, ["benchmark", "median"]]
    julia_versions = julia_versions[:, ["benchmark", "median"]]

    both = innerjoin(c_versions, julia_versions, on="benchmark", renamecols="_c"=>"_julia")
    both = both[both[!, "benchmark"] .!= "example", :]

    plot_julia_vs_c(both[:, "benchmark"], both[:, "median_c"], both[:, "median_julia"], version * (gpu ? " gpu" : " cpu"), preset)
end


# use the best version
function make_plot(preset::String, gpu::Bool)
    cd(@__DIR__)
    cd("..")
    df = CSV.read("results.csv", DataFrame)

    grouped = groupby(df, [:language, :preset, :gpu])

    c_versions = grouped[("C", preset, gpu)]
    julia_versions = grouped[("julia", preset, gpu)]

    c_grouped = groupby(c_versions, :benchmark)
    best_c = combine(c_grouped, :median => minimum => :best)
    c_versions = innerjoin(c_versions, best_c, on=["benchmark", "median" => "best"]) # a bit hacky...

    julia_grouped = groupby(julia_versions, :benchmark)
    best_julia = combine(julia_grouped, :median => minimum => :best)
    julia_versions = innerjoin(julia_versions, best_julia, on=["benchmark", "median" => "best"])

    c_versions = c_versions[:, ["benchmark", "median",  "version"]]
    julia_versions = julia_versions[:, ["benchmark", "median", "version"]]

    both = innerjoin(c_versions, julia_versions, on="benchmark", renamecols="_c"=>"_julia")
    both = both[both[!, "benchmark"] .!= "example", :]

    # to include also the version: both[:, "benchmark"] .* "\n(" .* both[:, "version_julia"] .* ")"

    plot_julia_vs_c(both[:, "benchmark"], both[:, "median_c"], both[:, "median_julia"], "best" * (gpu ? " gpu" : " cpu"), preset)

end
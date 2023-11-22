
include("timing/collect_measurements.jl")



function get_presets(args)
    id = findfirst(startswith("-p"), args)
    if isnothing(id)
        return ["missing", "S"] # by default only run S
    end
    ps = lowercase(args[id])[3:end]
    presets = ["missing"] # versions where no preset is specified are always run
    if contains(ps, "s") push!(presets, "S") end
    if contains(ps, "m") push!(presets, "M") end
    if contains(ps, "l") push!(presets, "L") end
    if contains(ps, "p") push!(presets, "paper") end
    return presets
end

function get_languages(args)
    id = findfirst(startswith("-l"), args)
    if isnothing(id)
        return [:julia, :C] # default no python
    end
    ls = lowercase(args[id])[3:end]
    langs = []
    if contains(ls, "j") push!(langs, :julia)  end
    if contains(ls, "c") push!(langs, :C)      end
    if contains(ls, "p") push!(langs, :python) end
    return langs
end

function get_benchmarks(args)
    bms = get_benchmarks()
    id = findfirst(startswith("-b"), args)
    if !isnothing(id)
        arg = args[id]
        bm = arg[3:end]
        if bm ∈ bms
            return [bm]
        else 
            println("do not understand $arg, the benchmark $bm doesnt exist")
            return []
        end
    end
    id = findfirst(!startswith("-"), args)
    if !isnothing(id)
        bm = args[id]
        if bm ∈ bms
            return [bm]
        else 
            println("the benchmark $bm doesnt exist (benchmark has to come before the version)")
            return []
        end
    end
    return bms
end

function get_version(args)
    bm_id = findfirst(!startswith("-"), args)
    v_id = findlast(!startswith("-"), args)
    if !isnothing(v_id) && bm_id != v_id
        return args[v_id]
    end
    id = findfirst(startswith("-v"), args)
    if !isnothing(id)
        arg = args[id]
        return arg[3:end]
    end
    return nothing
end


# can pass name of benchmark to run as first argument, and version as second argument :o
# if only benchmark is specified, it will run all versions
# if nothing is specified, it will run all benchmarks
main(args) = collect_measurements(
    get_benchmarks(args), 
    get_languages(args), 
    get_presets(args), 
    get_version(args)
)

main(ARGS)


try
    using ProfileView # https://github.com/timholy/ProfileView.jl
catch e
    import Pkg
    Pkg.add("ProfileView")
    using ProfileView
end


function profile(bm::String, julia_version::String, preset::String)
    file = joinpath(@__DIR__, "..", "benchmarks", bm, julia_version * ".jl")
    global PRESETS_TO_RUN = ["S"]
    global PROFILING = false
    include(file) # warmup
    PRESETS_TO_RUN = [preset]
    PROFILING = true
    include(file)
    PROFILING = false
end

function profile_gpu(bm::String, julia_version::String, preset::String)
    file = joinpath(@__DIR__, "..", "benchmarks", bm, julia_version * ".jl")
    global PRESETS_TO_RUN = ["S"]
    global PROFILING_GPU = false
    include(file) # warmup
    PRESETS_TO_RUN = [preset]
    PROFILING_GPU = true
    include(file)
    PROFILING_GPU = false
end
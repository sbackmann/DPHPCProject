include("../../timing/dphpc_timing.jl")

libpath = joinpath(@__DIR__, "libtest.so")

sleep1() = @ccall libpath.sleep1()::Cint

is_valid() = @ccall(libpath.test()::Cint) == 4

function main()
    if !isfile(libpath)
        run(Cmd(`make _make_lib`, dir=@__DIR__))
    end

    if is_valid()
        @dphpc_time sleep1()
    end
end

main()
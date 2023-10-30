
note that python support is still in the works


to run all benchmarks, do:
```
julia run_benchmarks.jl
```

there are different command line arguments to control which benchmarks to run
if you want to specify a certain benchmark, use -b followed by the name of the benchmark, e.g.:
```
julia run_benchmarks.jl -bfibonacci
```

you can specify the version you want to run with -v:
```
julia run_benchmarks.jl -bfibonacci -vdynamic
```

when you want to specify the version, you also have to give the benchmark
you cannot (yet?) do
```
julia run_benchmarks.jl -vdynamic
```

other than that, you can mix and match all the arguments

when specifying benchmark and version, you can also simply do
when you omit the -b and -v, the only thing to keep in mind is that the benchmark has to come before the version
```
julia run_benchmarks.jl fibonacci
julia run_benchmarks.jl fibonacci naive
```

if you only want to run benchnarks in certain languages, use -l followed by the languages you want to run
the first letter will be enough thanks
(default is julia and C)
to run all benchmarks with julia and python, julia C and python, or only python respectively, do
```
julia run_benchmarks.jl -ljp
julia run_benchmarks.jl -ljcp
julia run_benchmarks.jl -lp
```

you can run only specific presets, use -p followed by the presets you want to run
the options here are "S", "M", "L" and "paper", e.g.:
```
julia run_benchmarks.jl -pSMLpaper
julia run_benchmarks.jl -pSM
julia run_benchmarks.jl -pL
```

There are parameters in the /timing/dphpc_timing.h header that you can tune to influence how long the whole ordeal is gonna take. You can adjust
```
#define MIN_RUNS 10  // do at least _ runs
#define MAX_RUNS 200 // do at most _ runs
#define MAX_TIME 2.0 // dont run for more than _ seconds if enough measurements where collected
```

please look at the files in the fibonacci benchmark to see how to use the macros and how to write the makefile

:)
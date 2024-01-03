#ifndef DPHPC_TIMEH
#define DPHPC_TIMEH

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#define MIN_RUNS 10  // do at least _ runs
#define MAX_RUNS 200 // do at most _ runs
#define MAX_TIME 5.0 // dont run for more than _ seconds if enough measurements where collected
#define TIMEOUT  60.0 // after _ many seconds, dont start a new run, even if not enough measurements were collected



#include "presets.h"
int should_run_preset(const char* p) {
    for (int i = 0; i < nr_presets_to_run; i++) {
        if (strcmp(p, presets_to_run[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

long time_ns() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec * 1000000000 + t.tv_nsec;
}

long time_since_ns(long t) {
    return time_ns() - t;
}

double ns_to_sec(long ns) {
    return ns * 1e-9;
}

double ns_to_ms(long ns) {
    return ns * 1e-6;
}

int cmp(const void* ap, const void* bp) {
    long a, b;
    a = *((long*) ap);
    b = *((long*) bp);
    if (a > b) return 1;
    if (a < b) return -1;
    return 0;
}

void sort(long* measurements, size_t n) {
    qsort(measurements, n, sizeof(long), cmp);
}

int lb95_idx(int n) {
    static int lb_ids[] = {
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 10, 
        10, 11, 11, 12, 12, 12, 13, 13, 14, 14, 15, 15, 15, 16, 16, 17, 17, 18, 18, 18, 19, 
        19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 24, 24, 25, 25, 25, 26
    };
    if (n < 6) {
        
        
        return 0;
    }
    if (n <= 70) {
        return lb_ids[n-6];
    }
    return (int) floor(0.5*n - 0.98*sqrt(n));
}

int ub95_idx(int n) {
    static int ub_ids[] = {
        5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 14, 15, 15, 16, 16, 17, 18, 19, 19, 
        20, 20, 21, 21, 22, 22, 23, 23, 24, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 
        31, 32, 33, 34, 34, 35, 35, 36, 36, 37, 38, 38, 39, 39, 39, 40, 40, 41, 42, 43, 43
    };
    if (n < 6) {
        return n-1;
    }
    if (n <= 70) {
        return ub_ids[n-6];
    }
    return (int) ceil(0.5*n + 1 + 0.98*sqrt(n));
}



void median_CI95(long* measurements, int n) {
    sort(measurements, n);
    long median_ns    = measurements[n / 2];
    long median_lb_ns = measurements[lb95_idx(n)];
    long median_ub_ns = measurements[ub95_idx(n)];
    printf("median_lb_ms=%f, median_ms=%f, median_ub_ms=%f, ", 
        ns_to_ms(median_lb_ns), ns_to_ms(median_ns), ns_to_ms(median_ub_ns)
    );
}



// with reset you can reinitialize the inputs as needed
// expr is the code that is timed
// pass the preset as a string, use "S", "M", "L" or "paper"

#define dphpc_time(expr)         dphpc_time2(,expr)
#define dphpc_time2(reset, expr) dphpc_time3(reset, expr, "missing")

#define dphpc_time3(reset, expr, preset) ({                     \
    if (should_run_preset(preset)) {                            \
    long measurements_ns[MAX_RUNS];                             \
    int _nr_runs = 0;                                           \
    long start_time = time_ns();                                \
    for (int _i = 0; _i < MIN_RUNS; _i++) {                     \
        reset;                                                  \
        long _t = time_ns();                                    \
        expr;                                                   \
        measurements_ns[_nr_runs++] = time_since_ns(_t);        \
        if (ns_to_sec(time_since_ns(start_time)) > TIMEOUT) {   \
            break;                                              \
        }                                                       \
    }                                                           \
    for (int _i = MIN_RUNS; _i < MAX_RUNS; _i++) {              \
        if (ns_to_sec(time_since_ns(start_time)) > MAX_TIME) {  \
            break;                                              \
        }                                                       \
        reset;                                                  \
        long _t = time_ns();                                    \
        expr;                                                   \
        measurements_ns[_nr_runs++] = time_since_ns(_t);        \
    }                                                           \
    printf("dphpcresult(nr_runs=%d, ", _nr_runs);               \
    median_CI95(measurements_ns, _nr_runs);/* prints the rest */\
    if (strcmp(preset, "missing") == 0) {                       \
        printf("preset=missing)\n");                            \
    } else {                                                    \
        printf("preset=\"%s\")\n", preset);                     \
    }                                                           \
    }                                                           \
})




#endif // DPHPC_TIMEH



#ifndef DPHPC_TIMEH
#define DPHPC_TIMEH

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define MIN_RUNS 10  // do at least _ runs
#define MAX_RUNS 200 // do at most _ runs
#define MAX_TIME 2.0 // dont run for more than _ seconds if enough measurements where collected



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
        1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 
        11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16, 16, 17, 17, 18, 18, 19, 19, 19, 
        20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 24, 25, 25, 26, 26, 26, 27
    };
    if (n < 6) {
        printf("need at least 6 measurements to be able to give confidence intervals!\n");
        exit(123);
    }
    if (n <= 70) {
        return lb_ids[n-5];
    }
    return (int) floor(0.5*n - 0.98*sqrt(n));
}

int ub95_idx(int n) {
    static int ub_ids[] = {
        6, 7, 7, 8, 9, 10, 10, 11, 11, 12, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 19, 20, 
        20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 
        32, 32, 33, 34, 35, 35, 36, 36, 37, 37, 38, 39, 39, 40, 40, 40, 41, 41, 42, 43, 44, 44
    };
    if (n < 6) {
        printf("need at least 6 measurements to be able to give confidence intervals!\n");
        exit(321);
    }
    if (n <= 70) {
        return ub_ids[n-5];
    }
    return (int) ceil(0.5*n + 1 + 0.98*sqrt(n));
}



void median_CI95(long* measurements, int n) {
    sort(measurements, n);
    long median_ns    = measurements[n / 2];
    long median_lb_ns = measurements[lb95_idx(n)];
    long median_ub_ns = measurements[ub95_idx(n)];
    printf("median_lb_ms=%f, median_ms=%f, median_ub_ms=%f)\n", 
        ns_to_ms(median_lb_ns), ns_to_ms(median_ns), ns_to_ms(median_ub_ns)
    );
}

#define dphpc_time(expr) ({                                     \
    long measurements_ns[MAX_RUNS];                             \
    int nr_runs = 0;                                            \
    time_t start_time = time_ns();                              \
    for (int i = 0; i < MIN_RUNS; i++) {                        \
        long t = time_ns();                                     \
        expr;                                                   \
        measurements_ns[nr_runs++] = time_since_ns(t);          \
    }                                                           \
    for (int i = MIN_RUNS; i < MAX_RUNS; i++) {                 \
        if (ns_to_sec(time_since_ns(start_time)) > MAX_TIME) {  \
            break;                                              \
        }                                                       \
        long t = time_ns();                                     \
        expr;                                                   \
        measurements_ns[nr_runs++] = time_since_ns(t);          \
    }                                                           \
    printf("dphpcresult(nr_runs=%d, ", nr_runs);                \
    median_CI95(measurements_ns, nr_runs); /* prints the rest */\
})




#endif // DPHPC_TIMEH



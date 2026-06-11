#ifndef D2FF_COMMON_HPP
#define D2FF_COMMON_HPP

/* Default parameters — overridden by Makefile -D flags */
#ifndef REPS
    #if defined(__CUDACC__)
        #define REPS 4096
    #else
        #define REPS 1024
    #endif
#endif

#ifndef UNROLL
    #define UNROLL 8
#endif

#ifndef THREADS_PER_BLOCK
    #define THREADS_PER_BLOCK 256
#endif

#ifndef NUM_BLOCKS
    #define NUM_BLOCKS 2048
#endif

#ifndef NUM_ITERATIONS
    #define NUM_ITERATIONS 10
#endif

#ifndef DEPTH
    #define DEPTH 1
#endif

#define TOTAL_THREADS (THREADS_PER_BLOCK * NUM_BLOCKS)

struct d2ff_result {
    double time_ms;
    double sample;
};

/* double -> fp32mp2 (d2ff) */
d2ff_result d2ff_run_std();
d2ff_result d2ff_run_opt();

/* fp32mp2 -> double (ff2d) */
d2ff_result ff2d_run_std();
d2ff_result ff2d_run_opt();

#endif /* D2FF_COMMON_HPP */

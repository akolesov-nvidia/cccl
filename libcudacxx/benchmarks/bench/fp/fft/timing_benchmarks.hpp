/*
    timing_benchmarks.hpp - Timing Utilities for FFT Benchmarks
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Timing utilities and benchmarking infrastructure for measuring FFT performance on CUDA GPUs.
*/

#ifndef __TIMING_BENCHMARKS_HPP__
#define __TIMING_BENCHMARKS_HPP__

#include "common_benchmarks.hpp"

#include <stddef.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

// Include CUDA headers when compiling with NVCC
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda.h>
#endif

// Macro to catch CUDA errors in CUDA runtime calls
#ifdef __CUDACC__
#define CSC(call)                                                     \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
#else
// Host-only versions (no-op for host compilation)
#define CSC(call) call
#define CHECK_LAUNCH_ERROR()
#endif

// Function to get the current time in seconds
double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

// Class to measure the time of a function
class Timer 
{
    public:
       Timer();
       virtual ~Timer();
       void start();
       float stop();
    private:
#ifdef __CUDACC__
       cudaEvent_t event_start, event_stop;
#else
       double start_time;
#endif
 };

// Constructor for the Timer class
Timer::Timer() 
{
#ifdef __CUDACC__
    CSC(cudaEventCreate(&event_start));
    CSC(cudaEventCreate(&event_stop));
#else
    start_time = 0.0;
#endif
 }

// Start the timer
void Timer::start() 
{
#ifdef __CUDACC__
    CSC(cudaEventRecord(event_start, 0));
#else
    start_time = second();
#endif
 }

// Stop the timer and return the elapsed time in milliseconds
float Timer::stop() 
{
#ifdef __CUDACC__
    float time;
    CSC(cudaEventRecord(event_stop, 0));
    CSC(cudaEventSynchronize(event_stop));
    CSC(cudaEventElapsedTime(&time, event_start, event_stop));
    return time;
#else
    double end_time = second();
    return (float)((end_time - start_time) * 1000.0); // Convert to milliseconds
#endif
 }

// Destructor for the Timer class
Timer::~Timer() 
{
#ifdef __CUDACC__
    CSC(cudaEventDestroy(event_start));
    CSC(cudaEventDestroy(event_stop));
#endif
 }

#endif // __TIMING_BENCHMARKS_HPP__

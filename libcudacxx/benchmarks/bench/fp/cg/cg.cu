
/*
 * cg.cu - Conjugate Gradient (CG) Benchmark (CUDA)
 *
 * This file implements the Conjugate Gradient method for solving large, sparse, symmetric positive-definite linear systems.
 * It is designed as a performance benchmark for double-precision and emulated floating-point (fp64emu) arithmetic on NVIDIA GPUs.
 *
 * Features:
 *   - Supports both native double and custom fp64emu types (see fpemu_type.h)
 *   - Configurable problem size, iteration count, and tolerance via macros or Makefile
 *   - CUDA kernels for vector operations and matrix-vector multiplication
 *   - Simple, self-contained benchmarking harness
 *
 * Usage:
 *   - Build and run via the provided Makefile (see Makefile for options)
 *   - Output includes convergence information and timing results
 *
 * Author: Andrei Kolesov
 * Date:   2025-01-15
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include "fpemu_type.h"

// Type alias for easy switching between data types
#ifdef NO_EMULATION
using data_type = double;
#else
using data_type = fpemu;
#endif

#define PROBLEM_SIZE 2048
#define MAX_ITER 3000
#define TOLERANCE 1e-4

// CUDA thread configuration
#ifndef DEFAULT_THREADS_PER_BLOCK
  #define DEFAULT_THREADS_PER_BLOCK 256
#endif
#define DEFAULT_GRID_SIZE_MULTIPLIER 1.0  // Multiplier for grid size calculation
#define MIN_BLOCKS 1                       // Minimum number of blocks
#define MAX_BLOCKS 65535                   // Maximum number of blocks (CUDA limit)

/**
 * CUDA error checking macro
 * 
 * Wraps CUDA calls and exits with error message if the call fails.
 * Usage: CUDA_OK(cudaMalloc(&ptr, size));
 */
#define CUDA_OK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(1); \
  } \
} while(0)

/**
 * Prints basic GPU device information
 * 
 * Displays the GPU name and handles multi-GPU systems by selecting device 0.
 * Output is minimal: just the GPU name in one line.
 * 
 * Side effects: Sets CUDA device 0 as active if multiple devices exist
 */
void print_gpu_info() 
{
    int deviceCount;
    CUDA_OK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) 
    {
        printf("No CUDA devices found\n");
        return;
    }
    
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0)); // Use device 0
    printf("GPU: %s\n", prop.name);
    
    // Set device 0 as default if multiple devices exist
    if (deviceCount > 1) 
    {
        printf("Using device 0 (found %d devices)\n", deviceCount);
    }
}

/**
 * CUDA kernel: Fill vector with constant value
 * 
 * Sets all elements of vector 'a' to the constant value 'v'.
 * 
 * @tparam T Data type (double, fpemu, etc.)
 * @param a Device pointer to output vector
 * @param n Number of elements in vector
 * @param v Constant value to fill vector with
 * 
 * Thread mapping: thread i processes element a[i]
 */
template<typename T>
__global__ void fill_kernel(T* a, int n, T v) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}

/**
 * CUDA kernel: Vector operation y = a*x + b*y
 * 
 * Performs the BLAS-like operation: y[i] = a * x[i] + b * y[i]
 * 
 * @tparam T Data type (double, fpemu, etc.)
 * @param n Number of elements in vectors
 * @param a Scalar multiplier for vector x
 * @param x Input vector (read-only)
 * @param b Scalar multiplier for vector y
 * @param y Input/output vector (read-write)
 * 
 * Thread mapping: thread i processes element i
 */
template<typename T>
__global__ void axpby_kernel(int n, T a, const T* __restrict__ x,
                             T b, T* __restrict__ y) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = own_dot(a,b,x[i],y[i]);
}

/**
 * CUDA kernel: Vector operation y += a*x
 * 
 * Performs the BLAS-like operation: y[i] += a * x[i]
 * 
 * @tparam T Data type (double, fpemu, etc.)
 * @param n Number of elements in vectors
 * @param a Scalar multiplier
 * @param x Input vector (read-only)
 * @param y Input/output vector (read-write)
 * 
 * Thread mapping: thread i processes element i
 */
template<typename T>
__global__ void axpy_kernel(int n, T a, const T* __restrict__ x,
                            T* __restrict__ y) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = own_mad(a,x[i],y[i]);
}

/**
 * CUDA kernel: Vector copy operation y = x
 * 
 * Copies elements from vector x to vector y.
 * 
 * @tparam T Data type (double, fpemu, etc.)
 * @param y Output vector (destination)
 * @param x Input vector (source, read-only)
 * @param n Number of elements to copy
 * 
 * Thread mapping: thread i copies element x[i] to y[i]
 */
template<typename T>
__global__ void copy_kernel(T* __restrict__ y, const T* __restrict__ x, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

/**
 * CUDA kernel: Sparse matrix-vector multiplication with 2D Laplacian
 * 
 * Computes y = A*x where A is the 5-point stencil Laplacian operator on an N×N grid.
 * The Laplacian operator implements: y[i,j] = 4*x[i,j] - x[i-1,j] - x[i+1,j] - x[i,j-1] - x[i,j+1]
 * 
 * @tparam T Data type (double, fpemu, etc.)
 * @param x Input vector (read-only)
 * @param y Output vector (write-only)
 * @param N Grid dimension (creates N×N grid)
 * 
 * Thread mapping: thread idx processes grid point (r,c) where r = idx/N, c = idx%N
 * Boundary handling: Neumann boundary conditions (no change at edges)
 */
template<typename T>
__global__ void spmv2d_laplacian(const T* __restrict__ x,
                                 T* __restrict__ y, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = N * N;
    if (idx >= n) return;
    int r = idx / N, c = idx % N;
    T val = T(4.0) * x[idx];
    if (r > 0)   val -= x[idx - N];
    if (r < N-1) val -= x[idx + N];
    if (c > 0)   val -= x[idx - 1];
    if (c < N-1) val -= x[idx + 1];
    y[idx] = val;
}

/**
 * CUDA kernel: Dot product with block-level shared memory reduction
 * 
 * Computes the dot product of vectors a and b using a two-stage reduction:
 * 1. Each thread computes partial sums across the vector
 * 2. Block-level reduction using shared memory and tree-based algorithm
 * 
 * @tparam T Data type (double, fpemu, etc.)
 * @param a First input vector (read-only)
 * @param b Second input vector (read-only)
 * @param blockSums Output array for block sums (one per block)
 * @param n Number of elements in vectors
 * 
 * Thread mapping: Each thread processes multiple elements with stride
 * Shared memory: Uses blockDim.x * sizeof(T) bytes per block
 * Output: blockSums[blockIdx.x] contains the dot product for this block
 */
template<typename T>
__global__ void dot_kernel(const T* __restrict__ a,
                           const T* __restrict__ b,
                           T* __restrict__ blockSums,
                           int n) 
{
    extern __shared__ T s[];
    T sum = T(0.0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum = own_mad( a[i],b[i],sum);
    s[threadIdx.x] = sum;
    __syncthreads();
    for(int offset = blockDim.x>>1; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) s[threadIdx.x] += s[threadIdx.x + offset];
        __syncthreads();
    }
    if(threadIdx.x==0) blockSums[blockIdx.x]=s[0];
}

/**
 * Host function: Dot product computation with device memory
 * 
 * Orchestrates the dot product computation by:
 * 1. Launching the dot_kernel on the GPU
 * 2. Copying block sums back to host
 * 3. Performing final reduction on CPU
 * 
 * @tparam T Data type (double, fpemu, etc.)
 * @param n Number of elements in vectors
 * @param d_a Device pointer to first vector
 * @param d_b Device pointer to second vector
 * @param d_block Device pointer to block sums array
 * @param numBlocks Number of CUDA blocks to launch
 * @param threadsPerBlock Number of threads per block
 * @return T Dot product result (scalar)
 * 
 * Memory management: Uses thread-local static allocation for host buffer
 * Performance: Optimized for multiple calls with same numBlocks
 */
template<typename T>
T dot_device(int n, const T* d_a, const T* d_b,
                  T* d_block, int numBlocks, int threadsPerBlock) 
{
    size_t shmem = threadsPerBlock * sizeof(T);
    dot_kernel<T><<<numBlocks, threadsPerBlock, shmem>>>(d_a,d_b,d_block,n);
    CUDA_OK(cudaGetLastError());
    static thread_local T* h_partials=nullptr;
    static thread_local int cap=0;
    if(cap<numBlocks){
        free(h_partials);
        h_partials = (T*)malloc(numBlocks*sizeof(T));
        cap = numBlocks;
    }
    CUDA_OK(cudaMemcpy(h_partials,d_block,numBlocks*sizeof(T),cudaMemcpyDeviceToHost));
    T s=T(0.0);
    for(int i=0;i<numBlocks;i++) s+=h_partials[i];
    return s;
}

/**
 * Main function: Conjugate Gradient benchmark
 * 
 * Implements the complete CG benchmark workflow:
 * 1. Parameter parsing and validation
 * 2. GPU device setup and memory allocation
 * 3. Problem initialization (2D Laplacian system)
 * 4. CG iteration loop with timing
 * 5. Results output and cleanup
 * 
 * Command line arguments:
 * @param argc Number of arguments
 * @param argv Argument array
 *   - argv[1]: Grid size N (default: PROBLEM_SIZE)
 *   - argv[2]: Maximum iterations (default: MAX_ITER)
 *   - argv[3]: Convergence tolerance (default: TOLERANCE)
 *   - argv[4]: CUDA threads per block (default: DEFAULT_THREADS_PER_BLOCK)
 *   - argv[5]: Grid size multiplier (default: DEFAULT_GRID_SIZE_MULTIPLIER)
 * 
 * @return 0 on success, 1 on error
 * 
 * Problem: Solves A*x = b where A is the 2D Laplacian operator on N×N grid
 * Algorithm: Conjugate Gradient method with residual recomputation
 * Benchmark: Measures performance in GFLOP/s and convergence behavior
 */
int main(int argc, char** argv)
{
    // Parse command line parameters with defaults
    int N         = (argc>1)? atoi(argv[1]) : PROBLEM_SIZE;   // grid N x N
    int maxIters  = (argc>2)? atoi(argv[2]) : MAX_ITER;
    float tol    = (argc>3)? atof(argv[3]) : TOLERANCE;
    
    // CUDA thread configuration parameters
    int threadsPerBlock = (argc>4)? atoi(argv[4]) : DEFAULT_THREADS_PER_BLOCK;
    float gridMultiplier = (argc>5)? atof(argv[5]) : DEFAULT_GRID_SIZE_MULTIPLIER;
    
    // Validate thread parameters
    if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
        fprintf(stderr, "Error: Threads per block must be between 1 and 1024, got %d\n", threadsPerBlock);
        return 1;
    }
    if (gridMultiplier <= 0.0 || gridMultiplier > 10.0) {
        fprintf(stderr, "Error: Grid multiplier must be between 0.1 and 10.0, got %.2f\n", gridMultiplier);
        return 1;
    }

    // Print GPU info
    print_gpu_info();

    const int n = N*N;
    
    // Calculate optimal grid dimensions early
    int optimalGridSize = (n + threadsPerBlock - 1) / threadsPerBlock;
    int actualGridSize = (int)(optimalGridSize * gridMultiplier);
    
    // Ensure grid size is within valid range
    actualGridSize = max(MIN_BLOCKS, min(MAX_BLOCKS, actualGridSize));
    
    const int numBlocks = actualGridSize;
    
    // Validate grid size
    if (numBlocks < MIN_BLOCKS || numBlocks > MAX_BLOCKS) {
        fprintf(stderr, "Error: Calculated grid size %d is outside valid range [%d, %d]\n", 
                numBlocks, MIN_BLOCKS, MAX_BLOCKS);
        return 1;
    }

    printf("CG FP64 benchmark: N=%d (n=%d), maxIters=%d, tol=%.3e\n", N,n,maxIters,tol);
    printf("CUDA config: threadsPerBlock=%d, numBlocks=%d, gridMultiplier=%.2f\n", 
           threadsPerBlock, numBlocks, gridMultiplier);
    printf("Grid optimization: optimal=%d, actual=%d (multiplier=%.2f)\n", 
           optimalGridSize, actualGridSize, gridMultiplier);
           
    #if !defined(NO_EMULATION)
    printf("Using FPEMU emulation (method: %s)\n", ABC(__METHOD__));
    #else
    printf("Using native fp64 double precision\n");
    #endif
    
    // Show usage if help requested
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("\nUsage: %s [N] [maxIters] [tolerance] [threadsPerBlock] [gridMultiplier]\n", argv[0]);
        printf("  N              = grid size (default: %d)\n", PROBLEM_SIZE);
        printf("  maxIters       = maximum iterations (default: %d)\n", MAX_ITER);
        printf("  tolerance      = convergence tolerance (default: %.1e)\n", TOLERANCE);
        printf("  threadsPerBlock = CUDA threads per block (default: %d, range: 1-1024)\n", DEFAULT_THREADS_PER_BLOCK);
        printf("  gridMultiplier = grid size multiplier (default: %.1f, range: 0.1-10.0)\n", DEFAULT_GRID_SIZE_MULTIPLIER);
        printf("\nExamples:\n");
        printf("  %s                    # Use all defaults\n", argv[0]);
        printf("  %s 1024               # N=1024, other defaults\n", argv[0]);
        printf("  %s 1024 1000 1e-6    # N=1024, maxIters=1000, tol=1e-6\n", argv[0]);
        printf("  %s 1024 1000 1e-6 128 2.0  # Also threadsPerBlock=128, gridMult=2.0\n", argv[0]);
        return 0;
    }

    data_type *d_x, *d_b, *d_r, *d_p, *d_Ap, *d_xtrue, *d_block;
    CUDA_OK(cudaMalloc(&d_x, n*sizeof(data_type)));
    CUDA_OK(cudaMalloc(&d_b, n*sizeof(data_type)));
    CUDA_OK(cudaMalloc(&d_r, n*sizeof(data_type)));
    CUDA_OK(cudaMalloc(&d_p, n*sizeof(data_type)));
    CUDA_OK(cudaMalloc(&d_Ap, n*sizeof(data_type)));
    CUDA_OK(cudaMalloc(&d_xtrue, n*sizeof(data_type)));
    CUDA_OK(cudaMalloc(&d_block, numBlocks*sizeof(data_type)));

    // x_true = 1, b = A * x_true
    fill_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(d_xtrue,n,data_type(1.0));
    CUDA_OK(cudaGetLastError());
    spmv2d_laplacian<data_type><<<actualGridSize,threadsPerBlock>>>(d_xtrue,d_b,N);
    CUDA_OK(cudaGetLastError());

    fill_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(d_x,n,data_type(0.0));
    copy_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(d_r,d_b,n);
    copy_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(d_p,d_r,n);

    data_type bnorm = sqrt(dot_device<data_type>(n,d_b,d_b,d_block,numBlocks,threadsPerBlock));
    if(bnorm==data_type(0.0)) bnorm=data_type(1.0);

    // Calculate flops using native integer arithmetic to avoid emulation differences
    const float flops_spmv = 5.0*N*N - 4.0*N;
    const float flops_per_iter = flops_spmv + 10.0*n + 2.0;
    
    cudaEvent_t evStart, evStop;
    CUDA_OK(cudaEventCreate(&evStart));
    CUDA_OK(cudaEventCreate(&evStop));
    spmv2d_laplacian<data_type><<<actualGridSize,threadsPerBlock>>>(d_p,d_Ap,N);
    CUDA_OK(cudaDeviceSynchronize());

    const int RECOMPUTE_EVERY = 50;

    data_type rr = dot_device<data_type>(n,d_r,d_r,d_block,numBlocks,threadsPerBlock);
    // data_type init_resid = sqrt(rr)/bnorm;

    CUDA_OK(cudaEventRecord(evStart));

    int it=0;
    for(; it<maxIters; ++it)
    {
        spmv2d_laplacian<data_type><<<actualGridSize,threadsPerBlock>>>(d_p,d_Ap,N);

        data_type pAp = dot_device<data_type>(n,d_p,d_Ap,d_block,numBlocks,threadsPerBlock);

        if(!(pAp>data_type(0.0)))
        {
            fprintf(stderr,"Non-positive pAp at iter %d: %g\n", it, (double)pAp);
            break;
        }
        data_type alpha = rr/pAp;

        axpy_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(n,alpha,d_p,d_x);
        axpy_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(n,-alpha,d_Ap,d_r);

        data_type rr_new = dot_device<data_type>(n,d_r,d_r,d_block,numBlocks,threadsPerBlock);

        if(it % RECOMPUTE_EVERY == 0)
        {
            spmv2d_laplacian<data_type><<<actualGridSize,threadsPerBlock>>>(d_x,d_Ap,N);
            copy_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(d_r,d_b,n);
            axpy_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(n,data_type(-1.0),d_Ap,d_r);
            rr_new = dot_device<data_type>(n,d_r,d_r,d_block,numBlocks,threadsPerBlock);
        }

        // removed the early exit to fix iterations count
        /*/
        if(sqrt(rr_new)/bnorm < tol)
        {
            rr = rr_new;
            ++it;
            break;
        }
        */

        data_type beta = rr_new/rr;
        axpby_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(n,data_type(1.0),d_r,beta,d_p);

        rr = rr_new;
    }

    CUDA_OK(cudaEventRecord(evStop));
    CUDA_OK(cudaEventSynchronize(evStop));

    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms,evStart,evStop));

    spmv2d_laplacian<data_type><<<actualGridSize,threadsPerBlock>>>(d_x,d_Ap,N);
    copy_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(d_r,d_b,n);
    axpy_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(n,data_type(-1.0),d_Ap,d_r);
    float rel_resid = sqrt(dot_device<data_type>(n,d_r,d_r,d_block,numBlocks,threadsPerBlock))/bnorm;

    copy_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(d_r,d_x,n);
    axpy_kernel<data_type><<<actualGridSize,threadsPerBlock>>>(n,data_type(-1.0),d_xtrue,d_r);
    float xt_norm  = sqrt(dot_device<data_type>(n,d_xtrue,d_xtrue,d_block,numBlocks,threadsPerBlock));
    float rel_err  = sqrt(dot_device<data_type>(n,d_r,d_r,d_block,numBlocks,threadsPerBlock))/(xt_norm?xt_norm:1.0);

    float secs = ms*(1.0e-3);
    float gflops = (it*flops_per_iter)/secs/1.0e9;
    

    printf("\nResults:\n");
    printf("  Final rel residual: %.7e (tol=%.7e): %s\n", rel_resid,tol,rel_resid<tol?"OK":"FAIL");
    printf("  Rel error vs x*=1 : %.7e\n", (float)rel_err);
    printf("  Time              : %.3f ms\n", ms);
    printf("  Perf              : %.2f GFLOP/s\n",
           gflops);

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_r); cudaFree(d_p);
    cudaFree(d_Ap); cudaFree(d_xtrue); cudaFree(d_block);
    return 0;
}


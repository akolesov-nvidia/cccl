/* Optimized fp32mp2->double conversion (integer bit ops, FPMP_OPTIMIZED_FPMP_TO_DOUBLE=1) */
#define FPMP_OPTIMIZED_FPMP_TO_DOUBLE 1
#define FF2D_KERNEL_NAME  ff2d_kernel_opt
#define FF2D_WRAPPER_NAME ff2d_run_opt
#include "ff2d_kernel.hpp"

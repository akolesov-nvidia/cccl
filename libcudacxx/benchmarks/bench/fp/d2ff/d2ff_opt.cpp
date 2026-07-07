/* Optimized double->fp32mp2 conversion (integer bit ops, CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP=1) */
#define CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP 1
#define D2FF_KERNEL_NAME  d2ff_kernel_opt
#define D2FF_WRAPPER_NAME d2ff_run_opt
#include "d2ff_kernel.hpp"

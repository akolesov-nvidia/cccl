#ifndef FPEMU_TYPE_H
#define FPEMU_TYPE_H

// Macro to convert a macro value to a string
#define STRINGIFY(x) #x
#define ABC(x) STRINGIFY(x)

#ifndef __METHOD__
  #define __METHOD__ def
#endif

#if defined(USE_FPMP)
  #define CCCL_FPMP_EXPLICIT_CASTS 1
  #include <cuda/fpmp>
  #include <cuda/fpmp_math>
  using fpemu = ::cuda::experimental::fpmp2<float, ::cuda::experimental::fpmp2_accuracy::__METHOD__>;
  #define EMULATED_PRECISION
  #define PRECISION_LABEL "fpmp (accuracy: " ABC(__METHOD__) ")"
#elif defined(USE_FLOAT)
  #define PRECISION_LABEL "float"
#elif defined(NO_EMULATION)
  #define PRECISION_LABEL "double"
#else
  #include <cuda/fpemu>
  using fpemu = ::cuda::experimental::fpemu<double, ::cuda::experimental::fpemu_accuracy::__METHOD__>;
  #define EMULATED_PRECISION
  #define PRECISION_LABEL "fpemu (accuracy: " ABC(__METHOD__) ")"
#endif

#endif // FPEMU_TYPE_H

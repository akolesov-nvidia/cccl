#ifndef FPEMU_TYPE_H
#define FPEMU_TYPE_H

// Macro to convert a macro value to a string
#define STRINGIFY(x) #x
#define ABC(x) STRINGIFY(x)

#ifndef __METHOD__
  #define __METHOD__ def
#endif

#if defined(USE_FPMP)
  #define FPMP_EXPLICIT_CASTS 1
  #include "fpmp.hpp"
  #include "fpmp_math.hpp"
  using fp64emu_t = __nv_fpmp2_t<float, fpmp::method::__METHOD__>;
  #define EMULATED_PRECISION
  #define PRECISION_LABEL "fpmp (method: " ABC(__METHOD__) ")"
#elif defined(USE_FLOAT)
  #define PRECISION_LABEL "float"
#elif defined(NO_EMULATION)
  #define PRECISION_LABEL "double"
#else
  #include "fpemu.hpp"
  using fp64emu_t = __nv_fp64emu_t<fpemu::method::__METHOD__>;
  #define EMULATED_PRECISION
  #define PRECISION_LABEL "fpemu (method: " ABC(__METHOD__) ")"
#endif

#endif // FPEMU_TYPE_H

#ifndef __TS_TYPES_HPP__
#define __TS_TYPES_HPP__

namespace ts
{
    enum struct rounding
    {
        def = 0,
        rn  = 0,
        rz  = 1,
        ru  = 2,
        rd  = 3,
    };

    enum struct accuracy
    {
        def = 0,
        cr  = 0,
        ha  = 1,
        la  = 2,
        native = 3,
    };

    enum struct range
    {
        def    = 0,
        full   = 0,
        finite = 1,
        ftz    = 2,
        normal = 3,
    };

    enum struct method
    {
        def      = 0,
        fast     = 1,
        accurate = 2,
    };

    enum struct timing
    {
        def        = 0,
        throughput = 0,
        latency    = 1,
    };

    enum struct stream
    {
        def       = 0,
        stdout    = 0,    
        stderr    = 1,
        file      = 2,
        null      = 3,
    };

    enum struct result
    {
        success      =  0,
        warning      =  1,    
        error        = -1,
        out_of_range = -2,
    };

    struct timing_results_t
    {
        double latency_tst;
        double latency_ref;
        double throughput_tst;
        double throughput_ref;

        double events_latency_tst;
        double events_latency_ref;
        double events_throughput_tst;
        double events_throughput_ref;
    };

    // Helper to map METHOD enum value from macro token (ts::method)
    // Accuracy tokens (high/mid/low; def aliases high) map onto the ts-local
    // reference selector ts::method (accurate=cr / def=ha / fast=la).
    #define TS_METHOD_ENUM_HELPER_high ts::method::accurate
    #define TS_METHOD_ENUM_HELPER_mid  ts::method::def
    #define TS_METHOD_ENUM_HELPER_low  ts::method::fast
    #define TS_METHOD_ENUM_HELPER_def  ts::method::accurate
    #define TS_METHOD_ENUM_HELPER_INDIRECT(m) TS_METHOD_ENUM_HELPER_##m
    #define TS_METHOD_ENUM_HELPER(m) TS_METHOD_ENUM_HELPER_INDIRECT(m)
    #define METHOD_ENUM_VALUE TS_METHOD_ENUM_HELPER(__METHOD__)

    struct config_t 
    {
        static constexpr ts::rounding rm            = ts::rounding::__ROUNDING__;
        // METHOD is the external parameter, derive ACC and RANGE internally
        static constexpr ts::method mthd            = METHOD_ENUM_VALUE;
        // Map method to accuracy and range for internal analysis code
        static constexpr ts::accuracy acc           = (mthd == ts::method::accurate) ? ts::accuracy::cr :
                                                       (mthd == ts::method::def)      ? ts::accuracy::ha :
                                                       (mthd == ts::method::fast)     ? ts::accuracy::la :
                                                       ts::accuracy::ha;
        // Range expectation for the reference/analysis.
        //   * Legacy packed API: def/fast flush (FTZ/DAZ) -> range::normal; accurate
        //     is full IEEE -> range::full.
        //   * Unpacked approach (__FPEMU_PACKED_VIA_UNPACKED__==1, i.e. the
        //     fp64emu_unpacked type and the packed-via-unpacked test build): the
        //     unpack/pack are fully accurate and method-INDEPENDENT, so every method
        //     is full-range (no FTZ/DAZ). accurate is bit-exact IEEE -> range::full;
        //     def/fast handle denormals correctly (checked) but keep the sanctioned
        //     inf/nan relaxations (fp32 fast inf-inf, overflow boundary) -> range::finite.
#if defined(__FPEMU_PACKED_VIA_UNPACKED__) && (__FPEMU_PACKED_VIA_UNPACKED__ == 1)
        static constexpr ts::range range            = (mthd == ts::method::accurate) ? ts::range::full :
                                                       (mthd == ts::method::def)      ? ts::range::finite :
                                                       (mthd == ts::method::fast)     ? ts::range::finite :
                                                       ts::range::finite;
#else
        static constexpr ts::range range            = (mthd == ts::method::accurate) ? ts::range::full :
                                                       (mthd == ts::method::def)      ? ts::range::normal :
                                                       (mthd == ts::method::fast)     ? ts::range::normal :
                                                       ts::range::normal;
#endif
        static constexpr ts::stream console         = ts::stream::__CONSOLE__;

        static constexpr double mean                = __MEAN__;
        static constexpr double stddev              = __STDDEV__;
        static constexpr double check_nan_payload   = __NAN_PAYLOAD__;
        static constexpr int check_zero_sign        = __ZERO_SIGN__;
        static constexpr uint64_t timing_iterations = __ITERATIONS__;
        static constexpr uint64_t function_repeats    = __REPEATS__; 
        static constexpr uint64_t print_limit       = __PRINT_LIMIT__;
        static constexpr uint64_t seed              = __SEED__;
        static constexpr int run_timing             = __RUN_TIMING__;
        
        static constexpr int64_t max_len            = __MAX_LEN__;
        static constexpr int64_t accuracy_len       = __ACCURACY_LEN__;
        static constexpr int64_t latency_len        = __LATENCY_LEN__;
        static constexpr int64_t throughput_len     = __THROUGHPUT_LEN__;
        static constexpr int64_t htr_len            = __HTR_LEN__;

        uint64_t num_threads;
        uint64_t num_blocks;
        uint64_t num_threads_per_block;
        uint64_t num_blocks_per_grid;

        double mask_value                           = 1.0;
        FILE *logfile = nullptr;

    };
    config_t config;

    enum struct dataset_type_t
    {
        zero    = 0,
        inf     = 1,
        nan     = 2,
        denorm  = 3,
        fixed   = 4,
        hard    = 5,    
        key     = 6,
        finite  = 7,
        normal  = 8,
        uniform = 9,
        gauss   = 10,
        mixed   = 11,
    };

    // Helper struct to determine number of arguments in operator()
    template<typename T> struct function_traits;
    // Handle const operator() with references
    template<typename Function, typename Ret, typename... Args>
    struct function_traits<Ret(Function::*)(Args...) const>{ static constexpr size_t num_args = sizeof...(Args);};
    // Handle non-const operator() with references
    template<typename Function, typename Ret, typename... Args>
    struct function_traits<Ret(Function::*)(Args...)>{static constexpr size_t num_args = sizeof...(Args);};

    // Base template for dataset_array_t
    template<typename EmuFunction, 
             typename NativeFunction, 
             size_t nargs = function_traits<decltype(&NativeFunction::operator())>::num_args>
    struct dataset_array_t;    
    
    struct dataset_t
    {
        dataset_type_t type;
        std::string    name;
        int64_t        accuracy_len;
        int64_t        throughput_len;
        int64_t        latency_len;
    }
    all_datasets[] = 
    {
        { dataset_type_t::fixed,   "Fixed",         -1,                  0,                     0 }, 
        { dataset_type_t::hard,    "Hard-to-Round", -1,                  0,                     0 },    
        { dataset_type_t::zero,    "Zero",          32,                  0,                     0 },
        { dataset_type_t::inf,     "Inf",           32,                  0,                     0 },
        { dataset_type_t::nan,     "NaN",           1024,                0,                     0 },
        { dataset_type_t::key,     "Key-point",     -1,                  0,                     0 },
        { dataset_type_t::denorm,  "Denormal",      config.accuracy_len, config.throughput_len, config.latency_len }, 
        { dataset_type_t::finite,  "Finite",        config.accuracy_len, config.throughput_len, config.latency_len }, 
        { dataset_type_t::normal,  "Normal",        config.accuracy_len, config.throughput_len, config.latency_len }, 
        { dataset_type_t::uniform, "Uniform",       config.accuracy_len, config.throughput_len, config.latency_len },
        { dataset_type_t::gauss,   "Gaussian",      config.accuracy_len, config.throughput_len, config.latency_len },
        { dataset_type_t::mixed,   "Mixed",         config.accuracy_len, config.throughput_len, config.latency_len },
    };
    
    // Unified dataset_array_t - always contains all 4 input arrays
    template<typename EmuFunction, typename NativeFunction, uint64_t NumArgs>
    struct dataset_array_t
    {
        using emu_inp_type    = typename EmuFunction::inp_type;
        using emu_out_type    = typename EmuFunction::out_type;

        using native_inp_type = typename NativeFunction::inp_type;
        using native_out_type = typename NativeFunction::out_type;

        emu_inp_type* emu_inp1;
        emu_inp_type* emu_inp2;
        emu_inp_type* emu_inp3;
        emu_inp_type* emu_inp4;

        native_inp_type* native_inp1;
        native_inp_type* native_inp2;
        native_inp_type* native_inp3;
        native_inp_type* native_inp4;

        emu_out_type* emu_out1;
        native_out_type* native_out1;
        native_out_type* native_ref1;

        uint64_t max_len;

        dataset_array_t(uint64_t max_len) : max_len(max_len)
        {
                MALLOC(emu_inp1, emu_inp_type, max_len);
                MALLOC(emu_inp2, emu_inp_type, max_len);
                MALLOC(emu_inp3, emu_inp_type, max_len);
                MALLOC(emu_inp4, emu_inp_type, max_len);

                MALLOC(native_inp1, native_inp_type, max_len);
                MALLOC(native_inp2, native_inp_type, max_len);
                MALLOC(native_inp3, native_inp_type, max_len);
                MALLOC(native_inp4, native_inp_type, max_len);

                MALLOC(emu_out1, emu_out_type, max_len);
                MALLOC(native_out1, native_out_type, max_len);
                MALLOC(native_ref1, native_out_type, max_len);
                
                // Initialize memory with loop. The emulated input/output types may
                // be a class (packed fp64emu_t or unpacked fp64emu_unpacked_t) with
                // no implicit int assignment, so zero them via a double conversion
                // (every emulated type provides a from-double constructor).
                for (uint64_t i = 0; i < max_len; i++) {
                    emu_inp1[i]    = (emu_inp_type)0.0;
                    emu_inp2[i]    = (emu_inp_type)0.0;
                    emu_inp3[i]    = (emu_inp_type)0.0;
                    emu_inp4[i]    = (emu_inp_type)0.0;
                    emu_out1[i]    = (emu_out_type)0.0;

                    native_inp1[i] = 0;
                    native_inp2[i] = 0;
                    native_inp3[i] = 0;
                    native_inp4[i] = 0;
                    native_out1[i] = 0;

                    native_ref1[i] = 0;
                }
        }
        ~dataset_array_t()
        {
                FREE(emu_inp1);
                FREE(emu_inp2);
                FREE(emu_inp3);
                FREE(emu_inp4);
                FREE(emu_out1);

                FREE(native_inp1);
                FREE(native_inp2);
                FREE(native_inp3);
                FREE(native_inp4);
                FREE(native_out1);

                FREE(native_ref1);
        }
    };

    // Helper struct to handle 2x32 bit words
    typedef struct __attribute__((aligned(8))) { uint32_t x[2];  } uint32x2_t;

} // end of namespace ts

#endif // __TS_TYPES_HPP__

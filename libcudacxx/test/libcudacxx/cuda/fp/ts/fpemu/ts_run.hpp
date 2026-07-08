#ifndef __TS_RUN_HPP__
#define __TS_RUN_HPP__

// Detect the unpacked emulated type. Its in-memory layout is
// { uint32_t sign; uint32_t exponent; uint64_t mantissa; } (16 bytes), so the
// generic 8-byte dword XOR used to break compiler optimization on binary64 /
// packed operands would (a) be size-mismatched and (b) only touch sign+exponent,
// never the mantissa that actually feeds the unpacked cores. The timing-loop
// perturbation below special-cases this type to mutate the MANTISSA instead.
template<typename T> struct ts_is_unpacked { static constexpr bool value = false; };
template<fpemu_accuracy m> struct ts_is_unpacked<fpemu_unpacked<double, m>> { static constexpr bool value = true; };

// Function to mask the subnormal values
template<typename T, bool ftz = false>
__HOST_DECL__ __DEVICE_DECL__ T ftz_mask(T x)
{
    if constexpr (ftz)
    {
        uint64_t ix = ts::bit_cast<uint64_t>(x);
        ix = (((ix >> 52) & 0x7ff) == 0x0ull) ? ix & 0x8000000000000000ull : ix;
        return ts::bit_cast<T>(ix);
    }
    else
    {
        return x;
    }
}

// Unified accuracy kernel for 1-4 arguments
template<typename Function, typename Tin, typename Tout, bool ftz = false>
__GLOBAL_DECL__ void accuracy_kernel(Tin *arg1, Tin *arg2, Tin *arg3, Tin *arg4, Tout *res, uint64_t len)
{
    Function function;
    constexpr int num_args = ts::function_traits<decltype(&Function::operator())>::num_args;
    
#if defined __CUDACC__
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) 
#else
    for (int i = 0; i < len; i++)
#endif
    {
        if      constexpr (num_args == 1) {res[i] = ftz_mask<Tout, ftz>(ftz_mask<Tin, ftz>(function(arg1[i])));}
        else if constexpr (num_args == 2) {res[i] = ftz_mask<Tout, ftz>(ftz_mask<Tin, ftz>(function(ftz_mask<Tin, ftz>(arg1[i]), ftz_mask<Tin, ftz>(arg2[i]))));}
        else if constexpr (num_args == 3) {res[i] = ftz_mask<Tout, ftz>(ftz_mask<Tin, ftz>(function(ftz_mask<Tin, ftz>(arg1[i]), ftz_mask<Tin, ftz>(arg2[i]), ftz_mask<Tin, ftz>(arg3[i]))));}
        else if constexpr (num_args == 4) {res[i] = ftz_mask<Tout, ftz>(ftz_mask<Tin, ftz>(function(ftz_mask<Tin, ftz>(arg1[i]), ftz_mask<Tin, ftz>(arg2[i]), ftz_mask<Tin, ftz>(arg3[i]), ftz_mask<Tin, ftz>(arg4[i]))));}
    }
}

// Unified timing kernel for 1-4 arguments
template<typename Function, typename Tin, typename Tout>
__GLOBAL_DECL__ void timing_kernel(Tin *arg1, Tin *arg2, Tin *arg3, Tin *arg4, Tout *res, uint64_t len)
{
    Function function;
    constexpr int num_args = ts::function_traits<decltype(&Function::operator())>::num_args;
    
#if defined __CUDACC__
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) 
    {
        Tin  a1 = arg1[i];
        Tin  a2 = arg2[i];
        Tin  a3 = arg3[i];
        Tin  a4 = arg4[i];
        Tout r1;
        #pragma unroll (8)
        for (int r = 0; r < ts::config.function_repeats; r++)
        {
                 if constexpr (num_args == 1) { r1 = function(a1); }
            else if constexpr (num_args == 2) { r1 = function(a1, a2); }
            else if constexpr (num_args == 3) { r1 = function(a1, a2, a3); }
            else if constexpr (num_args == 4) { r1 = function(a1, a2, a3, a4); }

            // Artificial dependency between r1 and a1 to avoid compiler
            // optimization (hoisting/constant-folding the repeat loop). The
            // dependency must alter the value that actually flows into the kernel
            // each iteration.
            if constexpr (ts_is_unpacked<Tin>::value)
            {
                // Unpacked format: perturb the MANTISSA (the field the cores
                // operate on). Toggling bits 0 and 32 of the 64-bit significand
                // from r1's mantissa keeps the operand finite while forcing a real
                // data dependency, mirroring the two-bit dword XOR used below for
                // packed/binary64 operands. sign/exponent are left untouched so the
                // operand never drifts into a special (inf/nan-magic) exponent.
                a1.bits.mantissa ^= (r1.bits.mantissa & UINT64_C(0x0000000100000001));
            }
            else
            {
                ts::uint32x2_t r1_dwords = ts::bit_cast<ts::uint32x2_t>(r1);
                ts::uint32x2_t a1_dwords = ts::bit_cast<ts::uint32x2_t>(a1);
                a1_dwords.x[0]           = a1_dwords.x[0] ^ (r1_dwords.x[0] & 0x1u); // change low 32 bits
                a1_dwords.x[1]           = a1_dwords.x[1] ^ (r1_dwords.x[1] & 0x1u); // change high 32 bits
                a1                       = ts::bit_cast<Tin>(a1_dwords);
            }
        }
        res[i] = r1;
    }
#else
    for (int i = 0; i < len; i++)
    {
        for (int r = 0; r < ts::config.function_repeats; r++)
        {
            if      constexpr (num_args == 1) res[i] = function(arg1[i]);
            else if constexpr (num_args == 2) res[i] = function(arg1[i], arg2[i]);
            else if constexpr (num_args == 3) res[i] = function(arg1[i], arg2[i], arg3[i]);
            else if constexpr (num_args == 4) res[i] = function(arg1[i], arg2[i], arg3[i], arg4[i]);
        }
    }
#endif
}

template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ void run_accuracy(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, ts::dataset_t dataset)
{
    using emu_inp_type    = typename EmuFunction::inp_type;
    using emu_out_type    = typename EmuFunction::out_type;
    using native_inp_type = typename NativeFunction::inp_type;
    using native_out_type = typename NativeFunction::out_type;

    constexpr bool ftz = (ts::config.range == ts::range::ftz);

    uint64_t len = dataset.accuracy_len;

#if defined __CUDACC__
    int threadsPerBlock = __THREADS_PER_BLOCK__;
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    accuracy_kernel<EmuFunction, emu_inp_type, emu_out_type><<<blocksPerGrid, threadsPerBlock>>>(dataset_array.emu_inp1,    
                                                                                                 dataset_array.emu_inp2,    
                                                                                                 dataset_array.emu_inp3,    
                                                                                                 dataset_array.emu_inp4,    
                                                                                                 dataset_array.emu_out1, len);
    accuracy_kernel<NativeFunction, native_inp_type, native_out_type, ftz><<<blocksPerGrid, threadsPerBlock>>>(dataset_array.native_inp1, 
                                                                                                               dataset_array.native_inp2, 
                                                                                                               dataset_array.native_inp3, 
                                                                                                               dataset_array.native_inp4, 
                                                                                                               dataset_array.native_ref1, len);

    cudaDeviceSynchronize();
    CHECK_LAUNCH_ERROR();
#else // CPU
    accuracy_kernel<EmuFunction,    emu_inp_type,    emu_out_type>        (dataset_array.emu_inp1,    
                                                                           dataset_array.emu_inp2,    
                                                                           dataset_array.emu_inp3,    
                                                                           dataset_array.emu_inp4,    
                                                                           dataset_array.emu_out1, len);
    accuracy_kernel<NativeFunction, native_inp_type, native_out_type, ftz>(dataset_array.native_inp1, 
                                                                           dataset_array.native_inp2, 
                                                                           dataset_array.native_inp3, 
                                                                           dataset_array.native_inp4, 
                                                                           dataset_array.native_ref1, len);
#endif

    // Convert the emu outputs to the native outputs
    convert_outputs<EmuFunction, NativeFunction>(dataset_array, len);
}

template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ ts::timing_results_t run_timing(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, ts::dataset_t dataset)
{
    using emu_inp_type    = typename EmuFunction::inp_type;
    using emu_out_type    = typename EmuFunction::out_type;
    using native_inp_type = typename NativeFunction::inp_type;
    using native_out_type = typename NativeFunction::out_type;

    uint64_t throughput_len = (dataset.throughput_len == 0) ? ts::config.throughput_len : dataset.throughput_len;
    uint64_t latency_len    = (dataset.latency_len    == 0) ? ts::config.latency_len    : dataset.latency_len;

    float events, minevents = FLT_MAX;
    ts::timing_results_t results;

#if defined __CUDACC__ // GPU
    // Select GPU to run on
    struct cudaDeviceProp props;
    CSC (cudaSetDevice (0));
    CSC (cudaGetDeviceProperties (&props, 0));

    int latency_tpb    = 1;  
    int latency_bpg    = 1;
    int throughput_tpb =  __THREADS_PER_BLOCK__;
    int throughput_bpg = (throughput_len + throughput_tpb - 1) / throughput_tpb;
    int clockRate;

    CSC (cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0));

    Timer timer = Timer();

    minevents = FLT_MAX;
    for (int k = 0; k < (int)(ts::config.timing_iterations); k++)
    {
        timer.start(); // <<<<<<
        timing_kernel<EmuFunction, emu_inp_type, emu_out_type><<<throughput_bpg, throughput_tpb>>>(dataset_array.emu_inp1, 
                                                                                                   dataset_array.emu_inp2, 
                                                                                                   dataset_array.emu_inp3, 
                                                                                                   dataset_array.emu_inp4, 
                                                                                                   dataset_array.emu_out1, throughput_len);
        CHECK_LAUNCH_ERROR();
        events = timer.stop(); // <<<<<<
        if (events < minevents) minevents = events;
    }
    // Convert the emu outputs to the native outputs
    cudaDeviceSynchronize();
    convert_outputs<EmuFunction, NativeFunction>(dataset_array, throughput_len);

    // fake check to avoid compiler optimization
    if (dataset_array.native_out1[0] == static_cast<native_out_type>(1234567.1234567)) {
        native_out_type sum = 0; for (uint64_t i = 0; i < throughput_len; i++){ sum += dataset_array.native_out1[i];} printf("sum = %f\n", sum);}

    results.throughput_tst = (((double)ts::config.function_repeats * (double)(throughput_len))/
                             (props.multiProcessorCount))/
                             (double)(minevents * clockRate);
    results.events_throughput_tst = minevents;
    
    minevents = FLT_MAX;
    for (int k = 0; k < (int)(ts::config.timing_iterations); k++)
    {
        timer.start(); // <<<<<<
        timing_kernel<NativeFunction, native_inp_type, native_out_type><<<throughput_bpg, throughput_tpb>>>(dataset_array.native_inp1, 
                                                                                                            dataset_array.native_inp2, 
                                                                                                            dataset_array.native_inp3, 
                                                                                                            dataset_array.native_inp4, 
                                                                                                            dataset_array.native_out1, throughput_len);
        CHECK_LAUNCH_ERROR();
        events = timer.stop(); // <<<<<<
        if (events < minevents) minevents = events;
    }

    // fake check to avoid compiler optimization
    cudaDeviceSynchronize();
    if (dataset_array.native_out1[0] == static_cast<native_out_type>(1234567.1234567)){
        native_out_type sum = 0; for (uint64_t i = 0; i < throughput_len; i++){ sum += dataset_array.native_out1[i];} printf("sum = %f\n", sum);}

    results.throughput_ref = (((double)ts::config.function_repeats * (double)(throughput_len))/
                             (props.multiProcessorCount))/
                             (double)(minevents * clockRate);
    results.events_throughput_tst = minevents;
    minevents = FLT_MAX;
    for (int k = 0; k < (int)(ts::config.timing_iterations); k++)
    {
        timer.start(); // <<<<<<
        timing_kernel<EmuFunction, emu_inp_type, emu_out_type><<<latency_bpg, latency_tpb>>>(dataset_array.emu_inp1, 
                                                                                             dataset_array.emu_inp2, 
                                                                                             dataset_array.emu_inp3, 
                                                                                             dataset_array.emu_inp4, 
                                                                                             dataset_array.emu_out1, latency_len);
        CHECK_LAUNCH_ERROR();
        events = timer.stop(); // <<<<<<
        if (events < minevents) minevents = events;
    }
    // Convert the emu outputs to the native outputs
    cudaDeviceSynchronize();
    convert_outputs<EmuFunction, NativeFunction>(dataset_array, throughput_len);

    // fake check to avoid compiler optimization
    if (dataset_array.native_out1[0] == static_cast<native_out_type>(1234567.1234567)){
        native_out_type sum = 0; for (uint64_t i = 0; i < latency_len; i++){ sum += dataset_array.native_out1[i];} printf("sum = %f\n", sum);}

    results.latency_tst = (double)(minevents * clockRate) / 
                          (((double)ts::config.function_repeats * (double)(latency_len)));
    results.events_latency_tst = minevents;
    minevents = FLT_MAX;
    for (int k = 0; k < (int)(ts::config.timing_iterations); k++)
    {
        timer.start(); // <<<<<<
        timing_kernel<NativeFunction, native_inp_type, native_out_type><<<latency_bpg, latency_tpb>>>(dataset_array.native_inp1, 
                                                                                                      dataset_array.native_inp2, 
                                                                                                      dataset_array.native_inp3, 
                                                                                                      dataset_array.native_inp4, 
                                                                                                      dataset_array.native_out1, latency_len);
        CHECK_LAUNCH_ERROR();
        events = timer.stop(); // <<<<<<
        if (events < minevents) minevents = events;
    }
    // fake check to avoid compiler optimization
    cudaDeviceSynchronize();
    if (dataset_array.native_out1[0] == static_cast<native_out_type>(1234567.1234567)){
        native_out_type sum = 0; for (uint64_t i = 0; i < latency_len; i++){ sum += dataset_array.native_out1[i];} printf("sum = %f\n", sum); }

    results.latency_ref = (double)(minevents * clockRate) / 
                          (((double)ts::config.function_repeats * (double)(latency_len)));
    results.events_latency_ref = minevents;
#else // host
    double start_time, end_time, cur_time, min_time = DBL_MAX;
    double clock_rate = get_clock_rate();

    for (int k = 0; k < (int)(ts::config.timing_iterations); k++)
    {
        start_time = second(); // <<<<<<
        timing_kernel<EmuFunction, emu_inp_type, emu_out_type>(dataset_array.emu_inp1, 
                                                               dataset_array.emu_inp2, 
                                                               dataset_array.emu_inp3, 
                                                               dataset_array.emu_inp4, 
                                                               dataset_array.emu_out1, throughput_len);
        CHECK_LAUNCH_ERROR();
        end_time = second(); // <<<<<<
        cur_time = end_time - start_time;
        if (cur_time < min_time) min_time = cur_time;
    }
    results.throughput_tst = (((double)ts::config.function_repeats * (double)(throughput_len))/
                             (min_time * clock_rate));
    results.latency_tst = 1.0 / 
                          results.throughput_tst;

    results.events_throughput_tst = minevents;
    results.events_latency_tst = 1.0 / minevents;
    
    min_time = DBL_MAX;
    for (int k = 0; k < (int)(ts::config.timing_iterations); k++)
    {
        start_time = second(); // <<<<<<
        timing_kernel<NativeFunction, native_inp_type, native_out_type>(dataset_array.native_inp1, 
                                                                        dataset_array.native_inp2, 
                                                                        dataset_array.native_inp3, 
                                                                        dataset_array.native_inp4, 
                                                                        dataset_array.native_out1, throughput_len);
        CHECK_LAUNCH_ERROR();
        end_time = second(); // <<<<<<
        cur_time = end_time - start_time;
        if (cur_time < min_time) min_time = cur_time;
    }
    results.throughput_ref = (((double)ts::config.function_repeats * (double)(throughput_len))/
                               (min_time * clock_rate));
    results.latency_ref = 1.0 / 
                          results.throughput_ref;
    results.events_throughput_ref = minevents;
    results.events_latency_ref = 1.0 / minevents;

#endif
return results;
}

#endif // __TS_RUN_HPP__

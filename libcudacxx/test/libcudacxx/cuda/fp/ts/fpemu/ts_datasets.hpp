#ifndef __TS_DATASETS_HPP__
#define __TS_DATASETS_HPP__

// Key points for double precision floating point numbers
#define FP64_NEG_INF        0xfff0000000000000  // -Infinity
#define FP64_NEG_MAX_NORM   0xffefffffffffffff  // -1.79769e+308 (Most negative normalized)
#define FP64_NEG_ONE        0xbff0000000000000  // -1.0
#define FP64_NEG_MIN_NORM   0x8010000000000000  // -2.22507e-308 (Smallest negative normalized)
#define FP64_NEG_MAX_DENORM 0x800fffffffffffff  // -2.22507e-308 (Largest negative denormalized)
#define FP64_NEG_MIN_DENORM 0x8000000000000001  // -4.94066e-324 (Smallest negative denormalized)
#define FP64_NEG_ZERO       0x8000000000000000  // -0.0
#define FP64_POS_ZERO       0x0000000000000000  // +0.0
#define FP64_POS_MIN_DENORM 0x0000000000000001  // +4.94066e-324 (Smallest positive denormalized)
#define FP64_POS_MAX_DENORM 0x000fffffffffffff  // +2.22507e-308 (Largest positive denormalized)
#define FP64_POS_MIN_NORM   0x0010000000000000  // +2.22507e-308 (Smallest positive normalized)
#define FP64_POS_ONE        0x3ff0000000000000  // +1.0
#define FP64_POS_MAX_NORM   0x7fefffffffffffff  // +1.79769e+308 (Most positive normalized)
#define FP64_POS_INF        0x7ff0000000000000  // +Infinity
#define FP64_SNAN           0x7ff7ffffffffffff  // Signaling NaN
#define FP64_QNAN           0x7fffffffffffffff  // Quiet NaN

// Key points for double precision floating point numbers
static constexpr uint64_t keypoint[] = 
{
    FP64_NEG_INF,
    FP64_NEG_ONE,
    FP64_NEG_ZERO,
    FP64_POS_ZERO,
    FP64_POS_ONE,
    FP64_POS_INF,
    FP64_SNAN,
    FP64_QNAN,

    FP64_NEG_MAX_NORM,
    FP64_NEG_MIN_NORM,
    FP64_NEG_MAX_DENORM,
    FP64_NEG_MIN_DENORM,
    FP64_POS_MIN_DENORM,
    FP64_POS_MAX_DENORM,
    FP64_POS_MIN_NORM,
    FP64_POS_MAX_NORM,
}; 

static uint64_t keypoint_size = sizeof(keypoint) / sizeof(keypoint[0]);

// Function to get the gauss value
template<typename T>
__HOST_DECL__ T get_gauss_value(std::mt19937& gen, std::normal_distribution<double>& dist)
{
    return static_cast<T>(dist(gen));
}

// Function to get the uniform value
template<typename T>
__HOST_DECL__ T get_uniform_value(std::mt19937& gen, std::uniform_int_distribution<uint32_t>& sign_dist, 
                                                     std::uniform_int_distribution<int32_t>&  exp_dist, 
                                                     std::uniform_int_distribution<uint64_t>& mantissa_dist)
{
    uint64_t sign_field     = (uint64_t)sign_dist(gen);
    uint64_t mantissa_field = (uint64_t)mantissa_dist(gen);

    int64_t exp_unbiased    = exp_dist(gen);
    uint64_t exp_field      = (uint64_t)(exp_unbiased + _CCCL_FP64_BIAS);

    uint64_t res = (sign_field << 63 | ((exp_field << 52) & 0x7ff0000000000000) | (mantissa_field & 0xfffffffffffff));

    return ts::bit_cast<T>(res);
} 

// Function to get the key value
template<typename T>
__HOST_DECL__ T get_key_value(uint32_t& i)
{
    if (i >= keypoint_size) i = 0;
    return ts::bit_cast<T>(keypoint[i]);
}
 
// Function to get the mixed value
template<typename T>
__HOST_DECL__ T get_mixed_value(std::mt19937&                            gen,
                                std::uniform_int_distribution<uint32_t>& value_type_dist,
                                std::uniform_int_distribution<uint32_t>& key_dist, 
                                std::normal_distribution<double>&        normal_dist, 
                                std::uniform_int_distribution<uint32_t>& sign_dist,
                                std::uniform_int_distribution<int32_t>&  exp_dist,
                                std::uniform_int_distribution<uint64_t>& mantissa_dist)
{
    constexpr uint32_t gauss_threshold = 50;
    constexpr uint32_t uniform_threshold = 85;
    uint32_t key_idx    = key_dist(gen);
    uint32_t value_type = value_type_dist(gen);
    if      (value_type < gauss_threshold)   return get_gauss_value<T>(gen, normal_dist);
    else if (value_type < uniform_threshold) return get_uniform_value<T>(gen, sign_dist, exp_dist, mantissa_dist);
    else                                     return get_key_value<T>(key_idx);
}

// Function to fill the dataset array with the gauss dataset type
template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ uint64_t fill_dataset_gauss(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, 
                                          ts::dataset_t&                                    dataset)
{
    using native_inp_type = typename NativeFunction::inp_type;

    uint64_t len  = dataset_array.max_len;
    double mean   = ts::config.mean;
    double stddev = ts::config.stddev;

    std::mt19937 gen{ts::config.seed};  // Use fixed seed for reproducibility
    std::normal_distribution<double> dist{mean, stddev}; // Use provided mean and stddev

    for (uint64_t i = 0; i < len; i++) 
    {
        constexpr int num_args = ts::function_traits<decltype(&EmuFunction::operator())>::num_args;

                                     dataset_array.native_inp1[i] = get_gauss_value<native_inp_type>(gen, dist);
        if constexpr (num_args >= 2) dataset_array.native_inp2[i] = get_gauss_value<native_inp_type>(gen, dist);
        if constexpr (num_args >= 3) dataset_array.native_inp3[i] = get_gauss_value<native_inp_type>(gen, dist);
        if constexpr (num_args >= 4) dataset_array.native_inp4[i] = get_gauss_value<native_inp_type>(gen, dist);
    }

    if (dataset.accuracy_len == -1) dataset.accuracy_len = len;
    return dataset.accuracy_len;
}

// Function to fill the dataset array with the uniform dataset type
template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ uint64_t fill_dataset_uniform(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, 
                                            ts::dataset_t&                                    dataset, 
                                            int32_t                                           exp_beg = FP64_EXP_MIN, 
                                            int32_t                                           exp_end = FP64_EXP_MAX,
                                            uint64_t                                          mant_beg = 0,
                                            uint64_t                                          mant_end = 0xfffffffffffff,
                                            uint32_t                                          sign_beg = 0,
                                            uint32_t                                          sign_end = 1)
{
    using native_inp_type = typename NativeFunction::inp_type;

    uint64_t len = dataset_array.max_len;

    std::mt19937 gen{ts::config.seed};
    std::uniform_int_distribution<uint32_t> sign_dist    (sign_beg, sign_end);
    std::uniform_int_distribution<int32_t>  exp_dist     (exp_beg,  exp_end);
    std::uniform_int_distribution<uint64_t> mantissa_dist(mant_beg, mant_end);

    for (uint64_t i = 0; i < len; i++) 
    {
        constexpr int num_args = ts::function_traits<decltype(&EmuFunction::operator())>::num_args;

                                     dataset_array.native_inp1[i] = get_uniform_value<native_inp_type>(gen, sign_dist, exp_dist, mantissa_dist);
        if constexpr (num_args >= 2) dataset_array.native_inp2[i] = get_uniform_value<native_inp_type>(gen, sign_dist, exp_dist, mantissa_dist);
        if constexpr (num_args >= 3) dataset_array.native_inp3[i] = get_uniform_value<native_inp_type>(gen, sign_dist, exp_dist, mantissa_dist);
        if constexpr (num_args >= 4) dataset_array.native_inp4[i] = get_uniform_value<native_inp_type>(gen, sign_dist, exp_dist, mantissa_dist);

    }

    if (dataset.accuracy_len == -1) dataset.accuracy_len = len;
    return dataset.accuracy_len;
}

// Function to fill the dataset array with the key dataset type
template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ uint64_t fill_dataset_key(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, 
                                        ts::dataset_t&                                    dataset)
{
    using native_inp_type = typename NativeFunction::inp_type;

    uint64_t klen = 0;
    uint32_t i = 0, j = 0, k = 0, l = 0;
    uint64_t accuracy_len, max_len = dataset_array.max_len;

    constexpr int num_args = ts::function_traits<decltype(&NativeFunction::operator())>::num_args;

    if constexpr (num_args == 1) 
    {
        accuracy_len = MIN(max_len, keypoint_size);
        while (klen < max_len)
        {
            dataset_array.native_inp1[klen] = get_key_value<native_inp_type>(i);

            i++;
            klen++;
        }
    }
    else if constexpr (num_args == 2) 
    { 
        accuracy_len = MIN(max_len, keypoint_size * keypoint_size);
        while (klen < max_len)
        {
            dataset_array.native_inp1[klen] = get_key_value<native_inp_type>(i);
            dataset_array.native_inp2[klen] = get_key_value<native_inp_type>(j);

            i++;
            j = ((j+1) / keypoint_size);
            klen++;
        }
    }
    else if constexpr (num_args == 3) 
    {
        accuracy_len = MIN(max_len, keypoint_size * keypoint_size * keypoint_size);
        while (klen < max_len)
        {
            dataset_array.native_inp1[klen] = get_key_value<native_inp_type>(i);
            dataset_array.native_inp2[klen] = get_key_value<native_inp_type>(j);
            dataset_array.native_inp3[klen] = get_key_value<native_inp_type>(k);

            i++;
            j = ((j+1) / keypoint_size);
            k = ((k+1) / (keypoint_size * keypoint_size));
            klen++;
        }
    }
    else if constexpr (num_args == 4) 
    {
        accuracy_len = MIN(max_len, keypoint_size * keypoint_size * keypoint_size * keypoint_size);
        while (klen < max_len)
        {
            dataset_array.native_inp1[klen] = get_key_value<native_inp_type>(i);
            dataset_array.native_inp2[klen] = get_key_value<native_inp_type>(j);
            dataset_array.native_inp3[klen] = get_key_value<native_inp_type>(k);
            dataset_array.native_inp4[klen] = get_key_value<native_inp_type>(l);

            i++;
            j = ((j+1) / keypoint_size);
            k = ((k+1) / (keypoint_size * keypoint_size));
            l = ((l+1) / (keypoint_size * keypoint_size * keypoint_size));
            klen++;
        }
    }

    if (dataset.accuracy_len == -1) dataset.accuracy_len = accuracy_len;
    return dataset.accuracy_len;
}

// Function to fill the dataset array with the fixed dataset type
template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ uint64_t fill_dataset_fixed(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, 
                                          ts::dataset_t&                                    dataset)
{
    using native_inp_type = typename NativeFunction::inp_type;

    uint64_t len = dataset_array.max_len;
    uint64_t xlen = 0;

    constexpr int num_args = ts::function_traits<decltype(&NativeFunction::operator())>::num_args;

#if (defined __FIXED_INPUTS__)
    native_inp_type a1 = 0, a2 = 0, a3 = 0, a4 = 0;

    #if (defined (__A1__))
        std::string s1 = ABC(__A1__);
        a1 = ts::string_to_double(s1) * ts::config.mask_value; 
    #endif

    #if (defined (__A2__))
        std::string s2 = ABC(__A2__);
        a2 = ts::string_to_double(s2) * ts::config.mask_value;
    #endif

    #if (defined (__A3__))
        std::string s3 = ABC(__A3__);
        a3 = ts::string_to_double(s3) * ts::config.mask_value;
    #endif

    #if (defined (__A4__))
        std::string s4 = ABC(__A4__);
        a4 = ts::string_to_double(s4) * ts::config.mask_value;
    #endif

    if (!ts::config.run_timing) len = 1;
    for (xlen = 0; xlen < len; xlen++)
    {
                                     dataset_array.native_inp1[xlen] = a1;
        if constexpr (num_args >= 2) dataset_array.native_inp2[xlen] = a2;
        if constexpr (num_args >= 3) dataset_array.native_inp3[xlen] = a3;
        if constexpr (num_args >= 4) dataset_array.native_inp4[xlen] = a4;
    }

    dataset.throughput_len = ts::config.throughput_len;
    dataset.latency_len    = ts::config.latency_len;
#else
    uint64_t fixed_args = sizeof(ts::FIXED_NAME)/sizeof(ts::FIXED_NAME[0]);
    if (fixed_args)
    for (uint64_t i = 0; i < fixed_args; i++)
    {
        dataset_array.native_inp1[i] = ts::bit_cast<native_inp_type>(ts::FIXED_NAME[i].arg1);
        if constexpr (num_args >= 2) 
        {
            dataset_array.native_inp2[i] = ts::bit_cast<native_inp_type>(ts::FIXED_NAME[i].arg2);
        }
        if constexpr (num_args >= 3) 
        {
            dataset_array.native_inp3[i] = ts::bit_cast<native_inp_type>(ts::FIXED_NAME[i].arg3);
        }

        xlen++;
    }
#endif

    if (dataset.accuracy_len == -1) dataset.accuracy_len = xlen;
    return dataset.accuracy_len;
}

// Function to fill the dataset array with the hard-to-round dataset type
template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ uint64_t fill_dataset_hard(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, 
                                         ts::dataset_t&                                    dataset)
{

    using native_inp_type = typename NativeFunction::inp_type;
    using native_out_type = typename NativeFunction::out_type;

    uint64_t len = MIN(ts::config.htr_len, dataset_array.max_len);
    uint64_t htr_len = 0;

    std::mt19937 gen{ts::config.seed};
    std::uniform_int_distribution<int32_t>base_dist(-128, 128);
    std::uniform_int_distribution<int32_t> tweak_dist(1, 4);

    while (htr_len < len)
    {
        int32_t ia      = base_dist(gen);
        int32_t offset  = tweak_dist(gen);

        if constexpr (std::is_same_v<NativeFunction, add_function<native_inp_type,  native_out_type, ts::config.rm>> || 
                      std::is_same_v<NativeFunction, dadd_function<native_inp_type, native_out_type, ts::config.rm>> )
        {
            native_inp_type a  = (native_inp_type)ia/32.0;
            native_inp_type b = -a;
            uint64_t ib = ts::bit_cast<uint64_t>(b) + offset;
            dataset_array.native_inp1[htr_len] = a;
            dataset_array.native_inp2[htr_len] = ts::bit_cast<native_inp_type>(ib);
            htr_len++;
        }
        else if constexpr (std::is_same_v<NativeFunction, sub_function<native_inp_type,  native_out_type, ts::config.rm>> || 
                           std::is_same_v<NativeFunction, dsub_function<native_inp_type, native_out_type, ts::config.rm>> ||
                           std::is_same_v<NativeFunction, div_function<native_inp_type,  native_out_type, ts::config.rm>> || 
                           std::is_same_v<NativeFunction, ddiv_function<native_inp_type, native_out_type, ts::config.rm>>)
        {
            native_inp_type a  = (native_inp_type)ia/32.0;
            native_inp_type b = a;
            uint64_t ib = ts::bit_cast<uint64_t>(b) + offset;
            dataset_array.native_inp1[htr_len] = a;
            dataset_array.native_inp2[htr_len] = ts::bit_cast<native_inp_type>(ib);
            htr_len++;
        }
        else if constexpr (std::is_same_v<NativeFunction, mul_function <native_inp_type, native_out_type, ts::config.rm>> || 
                           std::is_same_v<NativeFunction, dmul_function<native_inp_type, native_out_type, ts::config.rm>>)
        {
            native_inp_type a  = (native_inp_type)ia/32.0;
            uint64_t ia = ts::bit_cast<uint64_t>(a)-1;
            uint64_t ib = ts::bit_cast<uint64_t>(a)+offset;
            dataset_array.native_inp1[htr_len] = ts::bit_cast<native_inp_type>(ia);
            dataset_array.native_inp2[htr_len] = ts::bit_cast<native_inp_type>(ib);
            htr_len++;
        }
        else if constexpr (std::is_same_v<NativeFunction, fma_function<native_inp_type,  native_out_type, ts::config.rm>> || 
                           std::is_same_v<NativeFunction, mad_function<native_inp_type,  native_out_type, ts::config.rm>> ||
                           std::is_same_v<NativeFunction, dfma_function<native_inp_type, native_out_type, ts::config.rm>>)
        {
            native_inp_type a  = (native_inp_type)ia/32.0;
            uint64_t ia = ts::bit_cast<uint64_t>(a)-1;
            uint64_t ib = ts::bit_cast<uint64_t>(a)+1;
            native_inp_type sign = (a > 0.0)?-1.0:1.0;
            native_inp_type c  = sign * ldexp(1.0, -53 + offset);
            dataset_array.native_inp1[htr_len] = ts::bit_cast<native_inp_type>(ia);
            dataset_array.native_inp2[htr_len] = ts::bit_cast<native_inp_type>(ib);
            dataset_array.native_inp3[htr_len] = c;
            htr_len++;
        }
        else
        {
            native_inp_type a  = (native_inp_type)ia/32.0;
            uint64_t ia = ts::bit_cast<uint64_t>(a)+offset;
            dataset_array.native_inp1[htr_len] = ts::bit_cast<native_inp_type>(ia);
            htr_len++;
        }
    } // end of while

    if (dataset.accuracy_len == -1) dataset.accuracy_len = htr_len;
    return dataset.accuracy_len;
}

// Function to fill the dataset array with the mixed dataset type
template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ uint64_t fill_dataset_mixed(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, 
                                          ts::dataset_t&                                    dataset,
                                          int32_t exp_min = FP64_EXP_MIN-1,
                                          int32_t exp_max = FP64_EXP_MAX+1)
{
    using native_inp_type = typename NativeFunction::inp_type;

    uint64_t len = dataset_array.max_len;

    std::mt19937 gen{ts::config.seed};
    std::uniform_int_distribution<uint32_t> sign_dist(0, 1);
    std::uniform_int_distribution<int32_t>  exp_dist(exp_min, exp_max);
    std::uniform_int_distribution<uint64_t> mantissa_dist(0, 0xfffffffffffff);
    std::normal_distribution<double>        normal_dist{ts::config.mean, ts::config.stddev};
    std::uniform_int_distribution<uint32_t> key_dist(0, keypoint_size-1);
    std::uniform_int_distribution<uint32_t> value_type_dist(0, 100);

    constexpr int num_args = ts::function_traits<decltype(&NativeFunction::operator())>::num_args;

    for (uint64_t i = 0; i < len; i++)
    {
                                      dataset_array.native_inp1[i] = get_mixed_value<native_inp_type>(gen, value_type_dist, key_dist, normal_dist, sign_dist, exp_dist, mantissa_dist);
        if constexpr (num_args >= 2)  dataset_array.native_inp2[i] = get_mixed_value<native_inp_type>(gen, value_type_dist, key_dist, normal_dist, sign_dist, exp_dist, mantissa_dist);
        if constexpr (num_args >= 3)  dataset_array.native_inp3[i] = get_mixed_value<native_inp_type>(gen, value_type_dist, key_dist, normal_dist, sign_dist, exp_dist, mantissa_dist);
        if constexpr (num_args >= 4)  dataset_array.native_inp4[i] = get_mixed_value<native_inp_type>(gen, value_type_dist, key_dist, normal_dist, sign_dist, exp_dist, mantissa_dist);
    }

    if (dataset.accuracy_len == -1) dataset.accuracy_len = len;
    return dataset.accuracy_len;
}

// Function to convert the native inputs to the emu inputs
template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ uint64_t convert_inputs_native_to_emu(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, uint64_t len )
{
    using emu_inp_type    = typename EmuFunction::inp_type;
    using native_inp_type = typename NativeFunction::inp_type;

    constexpr int num_args = ts::function_traits<decltype(&EmuFunction::operator())>::num_args;

    for (uint64_t i = 0; i < len; i++)                            { dataset_array.emu_inp1[i] = static_cast<emu_inp_type>(dataset_array.native_inp1[i]);}
    if constexpr (num_args >= 2){for (uint64_t i = 0; i < len; i++){dataset_array.emu_inp2[i] = static_cast<emu_inp_type>(dataset_array.native_inp2[i]);}}  
    if constexpr (num_args >= 3){for (uint64_t i = 0; i < len; i++){dataset_array.emu_inp3[i] = static_cast<emu_inp_type>(dataset_array.native_inp3[i]);}}
    if constexpr (num_args >= 4){for (uint64_t i = 0; i < len; i++){dataset_array.emu_inp4[i] = static_cast<emu_inp_type>(dataset_array.native_inp4[i]);}}

    return len;
}

// Function to convert the emu outputs to the native outputs
template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ uint64_t convert_outputs(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, uint64_t len )
{
    using native_out_type = typename NativeFunction::out_type;

    for (uint64_t i = 0; i < len; i++) {dataset_array.native_out1[i] = static_cast<native_out_type>(dataset_array.emu_out1[i]);}

    return len;
}

// Function to convert the reference outputs to the emu format and back
template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ uint64_t convert_refs(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, uint64_t len )
{
    using native_out_type = typename NativeFunction::out_type;
    using emu_out_type    = typename EmuFunction::out_type;

    for (uint64_t i = 0; i < len; i++) 
    {
        dataset_array.native_ref1[i] = static_cast<native_out_type>(static_cast<emu_out_type>(dataset_array.native_ref1[i]));
    }

    return len;
}

// Function to fill the dataset array with the current dataset type
template<typename EmuFunction, typename NativeFunction>
__HOST_DECL__ uint64_t fill_dataset(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, ts::dataset_t& dataset )
{
    uint64_t accuracy_len = 0;
    int64_t exp_min = FP64_EXP_MIN;
    int64_t exp_max = FP64_EXP_MAX;

    if (dataset.type == ts::dataset_type_t::key)
    {
        accuracy_len = fill_dataset_key<EmuFunction, NativeFunction>(dataset_array, dataset);
    }
    else if (dataset.type == ts::dataset_type_t::fixed)
    {
        accuracy_len = fill_dataset_fixed<EmuFunction, NativeFunction>(dataset_array, dataset);
    }       
    else if (dataset.type == ts::dataset_type_t::hard)
    {
        accuracy_len = fill_dataset_hard<EmuFunction, NativeFunction>(dataset_array, dataset);
    } 
    else if (dataset.type == ts::dataset_type_t::gauss)
    {
        accuracy_len = fill_dataset_gauss<EmuFunction, NativeFunction>(dataset_array, dataset);
    }
    else if (dataset.type == ts::dataset_type_t::zero)
    {
        accuracy_len = fill_dataset_uniform<EmuFunction, NativeFunction>(dataset_array, dataset, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1);
    }    
    else if (dataset.type == ts::dataset_type_t::denorm)
    {
        accuracy_len = fill_dataset_uniform<EmuFunction, NativeFunction>(dataset_array, dataset, exp_min-1, exp_min-1);
    }    
    else if (dataset.type == ts::dataset_type_t::uniform)
    {
        accuracy_len = fill_dataset_uniform<EmuFunction, NativeFunction>(dataset_array, dataset, exp_min-1, exp_max+1);
    }    
    else if (dataset.type == ts::dataset_type_t::normal)
    {
        accuracy_len = fill_dataset_uniform<EmuFunction, NativeFunction>(dataset_array, dataset, exp_min, exp_max);
    }
    else if (dataset.type == ts::dataset_type_t::finite)
    {
        accuracy_len = fill_dataset_uniform<EmuFunction, NativeFunction>(dataset_array, dataset, exp_min-1, exp_max);
    }
    else if (dataset.type == ts::dataset_type_t::inf)
    {
        accuracy_len = fill_dataset_uniform<EmuFunction, NativeFunction>(dataset_array, dataset, exp_max+1, exp_max+1, 0x0, 0x0, 0x0, 0x1);
    }   
    else if (dataset.type == ts::dataset_type_t::nan)
    {
        accuracy_len = fill_dataset_uniform<EmuFunction, NativeFunction>(dataset_array, dataset, exp_max+1, exp_max+1, 0x1, 0xfffffffffffff, 0x0, 0x1);
    }
    else if (dataset.type == ts::dataset_type_t::mixed)
    {
        accuracy_len = fill_dataset_mixed<EmuFunction, NativeFunction>(dataset_array, dataset, exp_min-1, exp_max+1);
    }

    // Convert the generated native inputs to the emu inputs
    convert_inputs_native_to_emu<EmuFunction, NativeFunction>(dataset_array, accuracy_len);

    return accuracy_len;
}

// Function to check if the dataset is included in the configuration
bool is_dataset_included(ts::dataset_t& dataset)
{
#if defined __FIXED_INPUTS__
    return (ts::dataset_type_t::fixed == dataset.type);
#else  
    constexpr uint64_t nd = sizeof(ts::all_datasets)/sizeof(ts::all_datasets[0]);
    ts::dataset_type_t included_datasets[nd];
    uint64_t d = 0;
  
    #if defined __DATASET_ZERO__
        included_datasets[d++] = ts::dataset_type_t::zero;
    #endif
    #if defined __DATASET_DENORM__
        included_datasets[d++] = ts::dataset_type_t::denorm;
    #endif
    #if defined __DATASET_KEY__
        included_datasets[d++] = ts::dataset_type_t::key;
    #endif
    #if defined __DATASET_FIXED__
        included_datasets[d++] = ts::dataset_type_t::fixed;
    #endif
    #if defined __DATASET_HARD__
        included_datasets[d++] = ts::dataset_type_t::hard;
    #endif
    #if defined __DATASET_GAUSS__
        included_datasets[d++] = ts::dataset_type_t::gauss;
    #endif
    #if defined __DATASET_UNIFORM__
        included_datasets[d++] = ts::dataset_type_t::uniform;
    #endif
    #if defined __DATASET_MIXED__
        included_datasets[d++] = ts::dataset_type_t::mixed;
    #endif
    #if defined __DATASET_NORMAL__
        included_datasets[d++] = ts::dataset_type_t::normal;
    #endif
    #if defined __DATASET_FINITE__
        included_datasets[d++] = ts::dataset_type_t::finite;
    #endif
    #if defined __DATASET_INF__
        included_datasets[d++] = ts::dataset_type_t::inf;
    #endif
    #if defined __DATASET_NAN__
        included_datasets[d++] = ts::dataset_type_t::nan;
    #endif

    if (d > 0)
    {
        for (uint64_t i = 0; i < d; i++)
        {
            if (dataset.type == included_datasets[i]) return true;
        }
        return false;
    }
    else
    {
        return true;
    }
#endif
}

// Function to check if the dataset accuracy length is greater than 0
bool run_accuracy_for_dataset(ts::dataset_t& dataset)   
{
    return dataset.accuracy_len > 0;
}

// Function to check if the dataset throughput and latency lengths are greater than 0
bool run_timing_for_dataset(ts::dataset_t& dataset)
{
    if(ts::config.run_timing)
    {
        return ((dataset.throughput_len > 0) && (dataset.latency_len > 0));
    }
    else
    {
        return false;
    }
}

#endif // __TS_DATASETS_HPP__

#ifndef __TS_ANALYZE_HPP__
#define __TS_ANALYZE_HPP__

template<typename T>
int32_t get_sign(T a)
{
    uint64_t a_bits = ts::bit_cast<uint64_t>(a);
    return (a_bits >> 63) & 0x1;
}

template<typename T>
int32_t get_exp(T a)
{
    uint64_t a_bits = ts::bit_cast<uint64_t>(a);
    return (a_bits >> 52) & 0x7FF;
}

template<typename T>
uint64_t get_mantissa(T a)
{
    uint64_t a_bits = ts::bit_cast<uint64_t>(a);
    return a_bits & 0xFFFFFFFFFFFFF;
}

template<typename T>
bool is_zero(T a)
{
    uint64_t a_bits = ts::bit_cast<uint64_t>(a);
    return (a_bits & (~(1ULL << 63))) == 0x0ul;
}

template<typename emu_out_type, typename Tref>
uint64_t compare_bits(emu_out_type a, Tref b)
{
    uint64_t a_bits = ts::bit_cast<uint64_t>(a);
    uint64_t b_bits = ts::bit_cast<uint64_t>(b);
    return (a_bits > b_bits)?(a_bits - b_bits):(b_bits - a_bits);
}

template<typename emu_out_type, typename Tref>
uint64_t compare_mantissa_bits(emu_out_type a, Tref b)
{
    uint64_t a_bits = get_mantissa(a);
    uint64_t b_bits = get_mantissa(b);
    return (a_bits > b_bits)?(a_bits - b_bits):(b_bits - a_bits);
}

template<typename emu_out_type, typename Tref>
uint32_t compare_exp_bits(emu_out_type a, Tref b)
{
    uint32_t a_exp = get_exp(a);
    uint32_t b_exp = get_exp(b);
    return (a_exp > b_exp)?(a_exp - b_exp):(b_exp - a_exp);
}

template<typename emu_out_type, typename Tref>
uint32_t compare_sign_bits(emu_out_type a, Tref b)
{
    uint32_t a_sign = get_sign(a);
    uint32_t b_sign = get_sign(b);
    return (a_sign > b_sign)?(a_sign - b_sign):(b_sign - a_sign);
}

ts::result check_diff(uint64_t diff)
{
    uint64_t threshold = 0;
    if (ts::config.acc == ts::accuracy::ha)      threshold = 1ull<<HA_ACC_THRESHOLD;
    else if (ts::config.acc == ts::accuracy::la) threshold = 1ull<<LA_ACC_THRESHOLD;
    else                                         threshold = 1ull<<CR_ACC_THRESHOLD;
    return (diff >= threshold)?ts::result::error : ts::result::success;
}

template<typename native_out_type, typename native_ref_type>
ts::result compare(native_out_type a, native_ref_type r, uint64_t& diff, int32_t& bits)
{
    bool fp32_mp_warning = false;
    (void)fp32_mp_warning; // suppress unused variable warning

    uint64_t a_bits     = ts::bit_cast<uint64_t>(a);
    uint64_t r_bits     = ts::bit_cast<uint64_t>(r);
    int32_t a_sign      = get_sign(a_bits);
    int32_t r_sign      = get_sign(r_bits);
    int32_t a_exp       = get_exp(a_bits);
    int32_t r_exp       = get_exp(r_bits);
    uint64_t a_mantissa = get_mantissa(a_bits);
    uint64_t r_mantissa = get_mantissa(r_bits);

    diff = compare_bits(a_bits, r_bits);
    frexp((double)diff, &bits);
    bits = (diff == 0)?0:bits;

    // Back to double
    double a_double = ts::bit_cast<double>(a_bits);
    double r_double = ts::bit_cast<double>(r_bits);

    // Branch for finite numbers
    if (std::isfinite(a_double) && std::isfinite(r_double))
    {
        // Branch for zeros and subnormals in ftz mode
        if (((ts::config.range == ts::range::ftz) || (ts::config.range == ts::range::normal)) && 
           (a_exp == 0 || r_exp == 0))
        {
            // Check if both are zeros (ftz reference)
            if (is_zero(a_bits) && is_zero(r_bits)) 
            {
                // Check if the sign is the same
                if (ts::config.check_zero_sign)
                {
                    if (a_sign == r_sign) { return  ts::result::success; }
                    else                  { bits = -1; return  ts::result::warning; }
                }
                else
                {
                    bits = 0;
                    return ts::result::success;
                }
            }
            else
            {
                // Check if the difference is within the threshold
                ts::result res = check_diff(diff);
                if (res == ts::result::error)
                {
                    bits = -1;
                    // Check if the reference result is a subnormal
                    if (((r_exp == 0) && (a_exp == 0)) || (fp32_mp_warning))
                    {
                        return ts::result::warning;
                    }
                    else
                    {
                        bits = 64;
                        return res;
                    }
                }
                else
                {
                    bits = (res == ts::result::warning)?-1:bits;
                    return res;
                }
                // If the difference is outside the threshold, return a warning
                res = (res == ts::result::error)?ts::result::warning : res;
                bits = (res == ts::result::warning)?-1:bits;
                return res;
            }
        } // End of branch for ftz
        else // Branch for normal numbers
        {
            // Exact zero on both sides: only the sign can differ. The sign of an
            // exact cancellation is rounding-dependent (IEEE-754 6.3: rd -> -0),
            // and the rounding-independent unpacked cores do not track it. Tolerate
            // a sign-only mismatch as a WARNING in directed rounding modes (rz/ru/rd)
            // while keeping round-to-nearest (rn) strict (FAIL).
            if (is_zero(a_bits) && is_zero(r_bits))
            {
                if (!ts::config.check_zero_sign || a_sign == r_sign)
                {
                    bits = 0;
                    return ts::result::success;
                }
                bits = -1;
                return (ts::config.rm == ts::rounding::rn) ? ts::result::error
                                                           : ts::result::warning;
            }
            // Return warning if fp32_mp numbers are close to denormal results
            if (fp32_mp_warning) { return ts::result::warning; }
            // Check if the difference is within the threshold
            return check_diff(diff);
        }    
    } // End of branch for finite numbers
    else // Branch for NaNs and infinities
    {
        // Branch for NaNs
        if (std::isnan(a_double) && std::isnan(r_double))
        {
            if (ts::config.check_nan_payload)
            {
                if (a_bits == r_bits) { bits = 0;  return  ts::result::success; }
                else                  { bits = -1; return  ts::result::warning; }
            }
            else
            {
                bits = 0; 
                return ts::result::success; 
            }
        }
        // Branch for infinities
        else if (std::isinf(a_double) && std::isinf(r_double))
        {
            if (a_sign == r_sign) { bits = 0; return  ts::result::success; }
            else                  { bits = -1; return  ts::result::error; }
        }
        // Branch for finite result and NaNs/infinities reference
        else if (std::isfinite(a_double) && (std::isinf(r_double) || std::isnan(r_double)))
        {
            if ((ts::config.range == ts::range::normal) || (ts::config.range == ts::range::finite))
            {
                bits = -1;
                return ts::result::warning;
            }
            else
            {
                bits = -1;
                return ts::result::error;
            }
        }
        // Branch for big finite reference and NaNs/INF result
        else if (std::isfinite(r_double) && (std::isinf(a_double)))
        {
            int32_t big_exp = FP64_EXP_MAX + _CCCL_FP64_BIAS;
            
            if ((ts::config.range == ts::range::normal) || (ts::config.range == ts::range::finite))
            {
                
                if (r_exp >= big_exp) { bits = -1; return ts::result::warning; }
                else                  { bits = -1; return ts::result::error; }
            }
            else
            {
                bits = -1;
                return ts::result::error;
            }
        }
        // Branch for other mixed types (INF vs NaN)
        else
        {
            bits = -1;
            if ((ts::config.range == ts::range::normal) || (ts::config.range == ts::range::finite))
            {
                return ts::result::warning;
            }
            else
            {
                return ts::result::error;
            }
        }
    } // End of branch for NaNs and infinities
} // End of compare

template<typename EmuFunction, typename NativeFunction>
int out_of_range(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, uint64_t idx)
{
    using emu_inp_type    = typename EmuFunction::inp_type;
    using emu_out_type    = typename EmuFunction::out_type;
    using native_out_type = typename NativeFunction::out_type;

    constexpr int num_args = ts::function_traits<decltype(&NativeFunction::operator())>::num_args;

    int is_not_finite = 0;
    int is_not_ftz    = 0;
    int is_not_normal = 0;

        double in1_double = ts::bit_cast<double>(dataset_array.emu_inp1[idx]);

        int in1_class =  std::fpclassify(in1_double);
        is_not_ftz    = is_not_ftz || (in1_class == FP_SUBNORMAL);
        is_not_finite = is_not_finite || (in1_class == FP_NAN) || (in1_class == FP_INFINITE);
        is_not_normal = is_not_normal || (in1_class == FP_SUBNORMAL) || ((in1_class == FP_NAN) || (in1_class == FP_INFINITE));

    if constexpr (num_args >= 2) 
    {
        double in2_double = ts::bit_cast<double>(dataset_array.emu_inp2[idx]);

        int in2_class = std::fpclassify(in2_double);
        is_not_ftz    = is_not_ftz || (in2_class == FP_SUBNORMAL);
        is_not_finite = is_not_finite || (in2_class == FP_NAN) || (in2_class == FP_INFINITE);
        is_not_normal = is_not_normal || (in2_class == FP_SUBNORMAL) || ((in2_class == FP_NAN) || (in2_class == FP_INFINITE));
    }
    if constexpr (num_args >= 3) 
    {
        double in3_double = ts::bit_cast<double>(dataset_array.emu_inp3[idx]);

        int in3_class = std::fpclassify(in3_double);
        is_not_ftz    = is_not_ftz || (in3_class == FP_SUBNORMAL);
        is_not_finite = is_not_finite || (in3_class == FP_NAN) || (in3_class == FP_INFINITE);
        is_not_normal = is_not_normal || (in3_class == FP_SUBNORMAL) || ((in3_class == FP_NAN) || (in3_class == FP_INFINITE));
    }
    if constexpr (num_args >= 4) 
    {
        double in4_double = ts::bit_cast<double>(dataset_array.emu_inp4[idx]);
        int in4_class = std::fpclassify(in4_double);
        is_not_ftz    = is_not_ftz || (in4_class == FP_SUBNORMAL);
        is_not_finite = is_not_finite || (in4_class == FP_NAN) || (in4_class == FP_INFINITE);
        is_not_normal = is_not_normal || (in4_class == FP_SUBNORMAL) || ((in4_class == FP_NAN) || (in4_class == FP_INFINITE));
    }

    if (ts::config.range == ts::range::ftz)
    {
        return is_not_ftz;
    }   
    else if (ts::config.range == ts::range::finite)
    {
        return is_not_finite;
    }   
    else if (ts::config.range == ts::range::normal)
    {
        return is_not_normal;
    }   
    else
    {
        return 0;
    }
}

template<typename EmuFunction, typename NativeFunction>
void analyze_results(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, ts::dataset_t dataset, int64_t& errors, int64_t& warnings, int32_t& bits_worst)
{
    using emu_inp_type    = typename EmuFunction::inp_type;
    using emu_out_type    = typename EmuFunction::out_type;
    using native_inp_type = typename NativeFunction::inp_type;
    using native_out_type = typename NativeFunction::out_type;
    using native_ref_type = typename NativeFunction::out_type;

    uint64_t prints = 0;
    uint64_t len = dataset.accuracy_len;
    uint64_t idx_worst = 0;
    uint64_t diff_worst = 0;
    ts::result res_worst = ts::result::success;

    errors     = 0;
    warnings   = 0;
    bits_worst = 0;
     
    for (uint64_t i = 0; i < len; i++)
    {
        uint64_t diff = 0;
        int32_t bits = -1;
        int is_out_of_range = out_of_range(dataset_array, i);
        ts::result res      = compare<native_out_type, native_ref_type>(dataset_array.native_out1[i], dataset_array.native_ref1[i], diff, bits);

        if (is_out_of_range && res == ts::result::error) { res = ts::result::warning; bits = -1; }

        if ((bits >= 0) && (bits > bits_worst))
        {
            bits_worst = bits;
            idx_worst  = i;
            diff_worst = diff;
            res_worst  = res;
        }

        warnings = (res == ts::result::warning)?warnings+1 : warnings;
        errors   = (res == ts::result::error)?errors+1 : errors;

        bool print_res = false;
        #if defined(__PRINT_OK__)
            print_res |= (res == ts::result::success);
        #endif
        #if defined(__PRINT_FAIL__)
            print_res |= (res == ts::result::error);
        #endif
        #if defined(__PRINT_WARN__)
            print_res |= (res == ts::result::warning);
        #endif
        #if defined(__PRINT_ALL__)
            print_res = true;
        #endif
        #if defined(__PRINT_NONE__)
            print_res = false;
        #endif
        
        if (print_res)
        {
            if (prints < ts::config.print_limit)
            {
                ts::printout(dataset_array, dataset, i, diff, bits, res);
                prints++;
            }
            else if (prints == ts::config.print_limit)
            {
                ts::printf_stream(ts::stream::stderr, "\n...\n\n\n");
                prints++;
            }
        }
    } // End of for (uint64_t i = 0; i < len; i++)

#if (!defined(__PRINT_OK__)) && (!defined(__PRINT_ALL__)) && (!defined(__PRINT_WARN__))
    if (errors > 0)
#endif
    {
        ts::printf_stream(ts::stream::stderr, ">>>>>>>>>>>>>> WORST CASE (%s)\n\n", dataset.name.c_str());
        ts::printout(dataset_array, dataset, idx_worst, diff_worst, bits_worst, res_worst);
        ts::printf_stream(ts::stream::stderr, "<<<<<<<<<<<<<<\n\n");
    }

    return;
}

#endif // __TS_ANALYZE_HPP__

#ifndef __TS_PRINT_HPP__
#define __TS_PRINT_HPP__

namespace ts
{

    __HOST_DECL__ __INLINE__ double string_to_double(const std::string& str)
    {
        char const* s = str.c_str();
        double d;
        if (strstr(s, "p"))
        {
            sscanf(s, "%la", &d);
        }
        else if(strstr(s, "0x"))
        {
            uint64_t d_bits64 = strtoull(s, nullptr, 16);
            d = ts::bit_cast<double>(d_bits64);
        }
        else
        {
            d = strtod(s, nullptr);
        }
        return d;
    }

    __HOST_DECL__ void printf_stream(stream s, const char* format, ...)
    {
        FILE *f = nullptr;

        if (s == stream::file)
        {
            if (ts::config.console == stream::file) { f = stdout; }
            else f = ts::config.logfile;
        }
        else if (s == stream::stderr)
        {
            if (ts::config.console == stream::stderr) { f = stdout; }
            else f = stderr;
        }
        else if (s == stream::stdout && ts::config.console == stream::null)
        {
            f = nullptr;
        }
        else
        {
            f = stdout;
        }

        if (f != nullptr)
        {
            va_list args;
            va_start(args, format);
            vfprintf(f, format, args);
            va_end(args);
        }
    } // end of printf_stream


    template <typename T>
    std::string to_string_prec(T value, int prec = 1) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(prec) << value;
        return oss.str();
    }

    std::string ll2str(uint64_t n)
    {
        if (n < 1000)
        {
            return std::to_string(n);
        }
        else if (n < 1000000)
        {
            double k = n / 1000.0;
            uint64_t r = (uint64_t)(n % 1000);
            return to_string_prec(k, (r >= 100)?1:0) + "K";
        }
        else
        {
            double m = n / 1000000.0;
            uint64_t r = ((uint64_t)n % 1000000);
            return to_string_prec(m, (r >= 100000)?1:0) + "M";
        }
    } // end of ll2str

    std::string print_results(int64_t errors, int64_t warnings, uint64_t len, int32_t bits)
    {
        char res_str[1024];
        int correct_bits = (bits <= 52)?52-bits:0;

        if (errors == -1 && warnings == -1)
        {
            snprintf(res_str, sizeof(res_str), "-- (%s)", ts::ll2str(0lu).c_str()); 
        }
        else if (errors == 0 && warnings == 0)
        {
            snprintf(res_str, sizeof(res_str), "OK   (%d bits/52: %s)", 
            correct_bits, ts::ll2str(len).c_str()); 
        }
        else if (errors > 0)
        {
            snprintf(res_str, sizeof(res_str), "FAIL (%d bits/52: %s/%s = %.0f%%)",
            correct_bits, 
                ts::ll2str(errors).c_str(), ts::ll2str(len).c_str(),
                std::ceil(100.0f*(double)errors / (double)len));
        }
        else if ((errors == 0) && (warnings > 0))
        {
            snprintf(res_str, sizeof(res_str), "WARN (%d bits/52: %s/%s = %.0f%%)", 
            correct_bits, 
                ts::ll2str(warnings).c_str(), ts::ll2str(len).c_str(),
                std::ceil(100.0f*(double)warnings / (double)len));
        }


        std::string ret_str;
        ret_str = res_str;
        return ret_str;
    } // end of print_results

    std::string ll2bin(uint64_t value) 
    {
        std::string result;
        for (int i = 63; i >= 0; --i) 
        {
            result += (value & (1ULL << i)) ? '1' : '0';
            if(i == 63 || i == 52) result += ':';
            //if(i == (64-12-23)) result += ':';
            if(i == 32) result += '\'';
        }
        return result;
    }

    template<typename EmuFunction, typename NativeFunction>
    void printout(ts::dataset_array_t<EmuFunction, NativeFunction>& dataset_array, ts::dataset_t dataset, uint64_t idx, uint64_t diff, int32_t diff_bits, ts::result res)
    {
        using EmuTin     = typename EmuFunction::inp_type;
        using EmuTout    = typename EmuFunction::out_type;
        using NativeTout = typename NativeFunction::out_type;

        constexpr int num_args = function_traits<decltype(&EmuFunction::operator())>::num_args;
        
        char out_str[1024];

        std::string res_str = (res == ts::result::success)? "OK" :(res == ts::result::error) ? "FAIL" : "WARNING";

        if constexpr (num_args == 4) 
        {
            snprintf(out_str, sizeof(out_str), "%s:\n%s<%s,%s>[%lu]\n\t"
            "arguments       = {0x%016lx, 0x%016lx, 0x%016lx, 0x%016lx} = {%.18lg, %.18lg, %.18lg, %.18lg} = {%la, %la, %la, %la}\n\t"
            "computed result = {0x%016lx} = {%-26.18lg} = {%-26la} = {%s}\n\t"
            "expected result = {0x%016lx} = {%-26.18lg} = {%-26la} = {%s}\n"
            "diff: %lu ~ %d bits (%s)\n\n", 
            dataset.name.c_str(),
            ABC(__FUNC__), ABC(__ROUNDING__), ABC(__METHOD__),
            idx,
            ts::bit_cast<uint64_t>(dataset_array.native_inp1[idx]),
            ts::bit_cast<uint64_t>(dataset_array.native_inp2[idx]), 
            ts::bit_cast<uint64_t>(dataset_array.native_inp3[idx]), 
            ts::bit_cast<uint64_t>(dataset_array.native_inp4[idx]), 
            (double)(dataset_array.native_inp1[idx]), 
            (double)(dataset_array.native_inp2[idx]), 
            (double)(dataset_array.native_inp3[idx]), 
            (double)(dataset_array.native_inp4[idx]), 
            (double)(dataset_array.native_inp1[idx]),  
            (double)(dataset_array.native_inp2[idx]), 
            (double)(dataset_array.native_inp3[idx]),
            (double)(dataset_array.native_inp4[idx]),
            ts::bit_cast<uint64_t>(dataset_array.native_out1[idx]), 
            (double)(dataset_array.native_out1[idx]), 
            (double)(dataset_array.native_out1[idx]), 
            ts::ll2bin(ts::bit_cast<uint64_t>(dataset_array.native_out1[idx])).c_str(),
            ts::bit_cast<uint64_t>(dataset_array.native_ref1[idx]), 
            (double)(dataset_array.native_ref1[idx]), 
            (double)(dataset_array.native_ref1[idx]), 
            ts::ll2bin(ts::bit_cast<uint64_t>(dataset_array.native_ref1[idx])).c_str(),
            diff, diff_bits, res_str.c_str());
        }       
        else if constexpr (num_args == 3) 
        {
            snprintf(out_str, sizeof(out_str), "%s:\n%s<%s,%s>[%lu]\n\t"
            "arguments       = {0x%016lx, 0x%016lx, 0x%016lx} = {%.18lg, %.18lg, %.18lg} = {%la, %la, %la}\n\t"
            "computed result = {0x%016lx} = {%-26.18lg} = {%-26la} = {%s}\n\t"
            "expected result = {0x%016lx} = {%-26.18lg} = {%-26la} = {%s}\n"
            "diff: %lu ~ %d bits (%s)\n\n", 
            dataset.name.c_str(),
            ABC(__FUNC__), ABC(__ROUNDING__), ABC(__METHOD__),
            idx,
            ts::bit_cast<uint64_t>(dataset_array.native_inp1[idx]),
            ts::bit_cast<uint64_t>(dataset_array.native_inp2[idx]), 
            ts::bit_cast<uint64_t>(dataset_array.native_inp3[idx]), 
            (double)(dataset_array.native_inp1[idx]), 
            (double)(dataset_array.native_inp2[idx]), 
            (double)(dataset_array.native_inp3[idx]), 
            (double)(dataset_array.native_inp1[idx]),  
            (double)(dataset_array.native_inp2[idx]), 
            (double)(dataset_array.native_inp3[idx]),
            ts::bit_cast<uint64_t>(dataset_array.native_out1[idx]), 
            (double)(dataset_array.native_out1[idx]), 
            (double)(dataset_array.native_out1[idx]), 
            ts::ll2bin(ts::bit_cast<uint64_t>(dataset_array.native_out1[idx])).c_str(),
            ts::bit_cast<uint64_t>(dataset_array.native_ref1[idx]), 
            (double)(dataset_array.native_ref1[idx]), 
            (double)(dataset_array.native_ref1[idx]), 
            ts::ll2bin(ts::bit_cast<uint64_t>(dataset_array.native_ref1[idx])).c_str(),
            diff, diff_bits, res_str.c_str());
        }
        else if constexpr (num_args == 2) 
        {
            snprintf(out_str, sizeof(out_str), "%s:\n%s<%s,%s>[%lu]\n\t"
            "arguments       = {0x%016lx, 0x%016lx}=={%.18lg, %.18lg}=={%la, %la}\n\t"
            "computed result = {0x%016lx}=={%-26.18lg}=={%-26la} = {%s}\n\t"
            "expected result = {0x%016lx}=={%-26.18lg}=={%-26la} = {%s}\n"
            "diff: %lu ~ %d bits (%s)\n\n", 
            dataset.name.c_str(),
            ABC(__FUNC__), ABC(__ROUNDING__), ABC(__METHOD__),
            idx,
            ts::bit_cast<uint64_t>(dataset_array.native_inp1[idx]),
            ts::bit_cast<uint64_t>(dataset_array.native_inp2[idx]), 
            (double)(dataset_array.native_inp1[idx]), 
            (double)(dataset_array.native_inp2[idx]), 
            (double)(dataset_array.native_inp1[idx]),  
            (double)(dataset_array.native_inp2[idx]), 
            ts::bit_cast<uint64_t>(dataset_array.native_out1[idx]), 
            (double)(dataset_array.native_out1[idx]), 
            (double)(dataset_array.native_out1[idx]), 
            ts::ll2bin(ts::bit_cast<uint64_t>(dataset_array.native_out1[idx])).c_str(),
            ts::bit_cast<uint64_t>(dataset_array.native_ref1[idx]), 
            (double)(dataset_array.native_ref1[idx]), 
            (double)(dataset_array.native_ref1[idx]),           
            ts::ll2bin(ts::bit_cast<uint64_t>(dataset_array.native_ref1[idx])).c_str(),
            diff, diff_bits, res_str.c_str());
        }
        else if constexpr (num_args == 1) 
        {
            snprintf(out_str, sizeof(out_str), "%s:\n%s<%s,%s>[%lu]\n\t"
            "argument        = {0x%016lx} = {%.18lg} = {%la}\n\t"
            "computed result = {0x%016lx} = {%-26.18lg} = {%-26la} = {%s}\n\t"
            "expected result = {0x%016lx} = {%-26.18lg} = {%-26la} = {%s}\n"
            "diff: %lu ~ %d bits (%s)\n\n", 
            dataset.name.c_str(),
            ABC(__FUNC__), ABC(__ROUNDING__), ABC(__METHOD__),
            idx,
            ts::bit_cast<uint64_t>(dataset_array.native_inp1[idx]), 
            (double)(dataset_array.native_inp1[idx]), 
            (double)(dataset_array.native_inp1[idx]), 
            ts::bit_cast<uint64_t>(dataset_array.native_out1[idx]), 
            (double)(dataset_array.native_out1[idx]), 
            (double)(dataset_array.native_out1[idx]),
            ts::ll2bin(ts::bit_cast<uint64_t>(dataset_array.native_out1[idx])).c_str(),
            ts::bit_cast<uint64_t>(dataset_array.native_ref1[idx]), 
            (double)(dataset_array.native_ref1[idx]), 
            (double)(dataset_array.native_ref1[idx]), 
            ts::ll2bin(ts::bit_cast<uint64_t>(dataset_array.native_ref1[idx])).c_str(),
            diff, diff_bits, res_str.c_str());
        }

        ts::printf_stream(ts::stream::stderr, "%s", out_str);
    } // end of printout

}

#endif
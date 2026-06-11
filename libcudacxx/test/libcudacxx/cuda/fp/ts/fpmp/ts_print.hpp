/*
    ts_print.hpp - FPMP Test Suite Printing Utilities
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Printing and formatting utilities for test output, including 128-bit integer support,
    floating-point value display, and accuracy error logging.
*/

#ifndef __TS_PRINT_HPP__
#define __TS_PRINT_HPP__


#include <cstring>

#include "ts_types.hpp"
#include "ts_utils.hpp"
#include "ts_dataset.hpp"
#include "ts_functions.hpp"

// ============================================================================
// Print a separator line (default: 84 dashes to match header width)
// ============================================================================
inline void print_separator(FILE* out, char ch = '-', int width = 84)
{
    for (int i = 0; i < width; i++) fputc(ch, out);
    fputc('\n', out);
}

namespace ts
{
    // ============================================================================
    // 128-bit unsigned integer type (for ll2bin)
    // ============================================================================
#if defined(__SIZEOF_INT128__)
    using uint128_t = __uint128_t;
#else
    // Fallback for platforms without native 128-bit int (e.g., MSVC, some CUDA)
    struct uint128_t {
        uint64_t lo, hi;
        uint128_t() : lo(0), hi(0) {}
        uint128_t(uint64_t v) : lo(v), hi(0) {}
        uint128_t operator|(const uint128_t& other) const { return uint128_t{lo | other.lo, hi | other.hi}; }
        uint128_t operator&(const uint128_t& other) const { return uint128_t{lo & other.lo, hi & other.hi}; }
        uint128_t operator<<(int shift) const {
            if (shift >= 64) return uint128_t{0, lo << (shift - 64)};
            if (shift == 0) return *this;
            return uint128_t{lo << shift, (hi << shift) | (lo >> (64 - shift))};
        }
        bool operator!=(uint64_t v) const { return lo != v || hi != 0; }
    private:
        uint128_t(uint64_t l, uint64_t h) : lo(l), hi(h) {}
    };
#endif

    // ============================================================================
    // String-to-value parsers (used by string_to_value for fixed input mode)
    // ============================================================================

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
            d = bit_cast<double>(d_bits64);
        }
        else
        {
            d = strtod(s, nullptr);
        }
        return d;
    }

    __HOST_DECL__ __INLINE__ float string_to_float(const std::string& str)
    {
        char const* s = str.c_str();
        float f;
        if (strstr(s, "p"))
        {
            sscanf(s, "%a", &f);
        }
        else if(strstr(s, "0x"))
        {
            uint64_t f_bits64 = strtoull(s, nullptr, 16);
            f = bit_cast<float>(bit_cast<uint32_t>(f_bits64));
        }
        else
        {
            f = strtof(s, nullptr);
        }
        return f;
    }

    __HOST_DECL__ __INLINE__ int32_t string_to_int32(const std::string& str)
    {    
        return static_cast<int32_t>(strtol(str.c_str(), nullptr, 0));
    }

    __HOST_DECL__ __INLINE__ uint32_t string_to_uint32(const std::string& str)
    {    
        return static_cast<uint32_t>(strtoul(str.c_str(), nullptr, 0));
    }

    __HOST_DECL__ __INLINE__ int64_t string_to_int64(const std::string& str)
    {    
        return strtoll(str.c_str(), nullptr, 0);
    }

    __HOST_DECL__ __INLINE__ uint64_t string_to_uint64(const std::string& str)
    {    
        return strtoull(str.c_str(), nullptr, 0);
    }

    // Parse a multi-precision value from a string with two separated components
    // Supported formats: "hi:lo", "hi,lo", "(hi: lo)", "(hi, lo)"
    template<typename TestType>
    __HOST_DECL__ __INLINE__ TestType string_to_mp(const std::string& str)
    {
        using ComponentType = mp_component_t<TestType>;
        
        // Find separator: prefer colon, fall back to comma
        size_t sep_pos = str.find(':');
        if (sep_pos == std::string::npos)
            sep_pos = str.find(',');
        
        if (sep_pos == std::string::npos)
        {
            // No separator found - treat as single value with lo = 0
            ComponentType hi;
            if constexpr (std::is_same_v<ComponentType, float>)
                hi = string_to_float(str);
            else
                hi = string_to_double(str);
            return TestType(hi, ComponentType(0));
        }
        
        // Extract and trim hi/lo parts
        std::string hi_str = str.substr(0, sep_pos);
        std::string lo_str = str.substr(sep_pos + 1);
        
        size_t hi_start = hi_str.find_first_not_of(" \t(");
        size_t hi_end = hi_str.find_last_not_of(" \t");
        if (hi_start != std::string::npos && hi_end != std::string::npos)
            hi_str = hi_str.substr(hi_start, hi_end - hi_start + 1);
        
        size_t lo_start = lo_str.find_first_not_of(" \t");
        size_t lo_end = lo_str.find_last_not_of(" \t)");
        if (lo_start != std::string::npos && lo_end != std::string::npos)
            lo_str = lo_str.substr(lo_start, lo_end - lo_start + 1);
        else
            lo_str.clear();
        
        ComponentType hi, lo;
        if constexpr (std::is_same_v<ComponentType, float>)
        {
            hi = string_to_float(hi_str);
            lo = lo_str.empty() ? ComponentType(0) : string_to_float(lo_str);
        }
        else
        {
            hi = string_to_double(hi_str);
            lo = lo_str.empty() ? ComponentType(0) : string_to_double(lo_str);
        }
        
        return TestType(hi, lo);
    }

    // Universal string-to-value parser (dispatches based on type)
    template<typename TestType>
    __HOST_DECL__ __INLINE__ TestType string_to_value(const std::string& str)
    {
        if constexpr (is_multiprecision_v<TestType>)
        {
            return string_to_mp<TestType>(str);
        }
        else if constexpr (std::is_same_v<TestType, float>)
        {
            return string_to_float(str);
        }
        else if constexpr (std::is_same_v<TestType, double>)
        {
            return string_to_double(str);
        }
        else if constexpr (std::is_integral_v<TestType>)
        {
            if constexpr (std::is_signed_v<TestType>)
            {
                if constexpr (sizeof(TestType) <= 4)
                    return static_cast<TestType>(string_to_int32(str));
                else
                    return static_cast<TestType>(string_to_int64(str));
            }
            else
            {
                if constexpr (sizeof(TestType) <= 4)
                    return static_cast<TestType>(string_to_uint32(str));
                else
                    return static_cast<TestType>(string_to_uint64(str));
            }
        }
        else
        {
            static_assert(sizeof(TestType) == 0, "Unsupported type for string_to_value");
            return TestType{};
        }
    }
   
    // ============================================================================
    // Stream output with redirection support
    // ============================================================================

    __HOST_DECL__ void printf_stream(stream s, const char* format, ...)
    {
        FILE *f = nullptr;

        if (s == stream::file)
        {
            if (console == stream::file) { f = stdout; }
            else f = logfile;
        }
        else if (s == stream::stderr)
        {
            if (console == stream::stderr) { f = stdout; }
            else f = stderr;
        }
        else if (s == stream::stdout && console == stream::null)
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
    }

    // ============================================================================
    // CSV log file header
    // ============================================================================
    __HOST_DECL__ void write_csv_header(FILE* f)
    {
        if (f != nullptr)
        {
            fprintf(f, "function,type,method,work_rel_err,work_bits,work_status,"
                       "base_gflops,test_gflops,ref_gflops,"
                       "base_ev_clk_sm,test_ev_clk_sm,ref_ev_clk_sm,ev_clk_ratio_base,ev_clk_ratio_ref,"
                       "base_clk_ev,test_clk_ev,ref_clk_ev,clk_ev_ratio_base,clk_ev_ratio_ref\n");
            fflush(f);
        }
    }

    // ============================================================================
    // Binary representation for debugging
    // ============================================================================

    // Convert 64-bit value to binary string (IEEE 754 binary64 / double format)
    // Format: sign:exponent:mantissa (1:11:52 bits)
    std::string ll2bin(uint64_t value) 
    {
        std::string result;
        for (int i = 63; i >= 0; --i) 
        {
            result += (value & (1ULL << i)) ? '1' : '0';
            if(i == 63 || i == 52) result += ':';  // sign:exp:mantissa separators
            if(i == 4) result += '\'';             // 48 mantissa bits boundary (fp32mp2)
        }
        return result;
    }

    // Convert 128-bit value to binary string (for fp64mp2)
    // Format: sign:exponent:mantissa (1:15:112 bits) with 106-bit marker
    std::string ll2bin(uint128_t value) 
    {
        std::string result;
#if defined(__SIZEOF_INT128__)
        for (int i = 127; i >= 0; --i) 
        { 
            result += (value & (static_cast<uint128_t>(1) << i)) ? '1' : '0'; 
            if(i == 127 || i == 112) result += ':';
            if(i == 6) result += '\'';  // 106 mantissa bits boundary
        }
#else
        // Struct fallback: hi contains bits 127-64, lo contains bits 63-0
        for (int i = 63; i >= 0; --i) 
        { 
            int bit_pos = i + 64;
            result += (value.hi & (1ULL << i)) ? '1' : '0'; 
            if(bit_pos == 127 || bit_pos == 112) result += ':';
        }
        for (int i = 63; i >= 0; --i) 
        { 
            result += (value.lo & (1ULL << i)) ? '1' : '0'; 
            if(i == 6) result += '\'';
        }
#endif
        return result;
    }

} // end of namespace ts

// Import commonly used items from ts namespace for cleaner code
using ts::bit_cast;
using ts::is_multiprecision_v;
using ts::mp_component_t;
using ts::uint128_t;
using ts::ll2bin;

// ============================================================================
// Accuracy Print Functions (depend on fpmp_type, fprf_type, accuracy types)
// ============================================================================

// ----------------------------------------------------------------------------
// IEEE 754 binary128 (%a-style) from raw bits — no libquadmath.
// Assumes in-memory layout matches GCC __float128 / 128-bit IEEE long double
// (LE: low 8 bytes = bits 0–63, high 8 bytes = bits 64–127; BE: halves swapped).
// Decimal "full" column uses double conversion only (human impression, not exact).
// ----------------------------------------------------------------------------
namespace ts
{
    inline void read_ieee_binary128_u64_pair(const void* ptr, uint64_t* hi_ieee, uint64_t* lo_ieee)
    {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        std::memcpy(hi_ieee, ptr, sizeof(uint64_t));
        std::memcpy(lo_ieee, static_cast<const char*>(ptr) + sizeof(uint64_t), sizeof(uint64_t));
#else
        std::memcpy(lo_ieee, ptr, sizeof(uint64_t));
        std::memcpy(hi_ieee, static_cast<const char*>(ptr) + sizeof(uint64_t), sizeof(uint64_t));
#endif
    }

    // One hex digit (0..27) of the 112-bit fraction field; j=0 is bits 111..108.
    inline unsigned ieee112_fraction_nibble(uint64_t frac_hi48, uint64_t frac_lo64, int j)
    {
        const int low_bit = 108 - 4 * j;
        if (low_bit >= 64)
            return static_cast<unsigned>((frac_hi48 >> (low_bit - 64)) & 0xFu);
        if (low_bit + 3 < 64)
            return static_cast<unsigned>((frac_lo64 >> low_bit) & 0xFu);
        const uint64_t w = (frac_lo64 >> low_bit) | (frac_hi48 << (64 - low_bit));
        return static_cast<unsigned>(w & 0xFu);
    }

    // Writes "+0x1.mmmmmmmmmmmmmmmmmmmmmmmmmmmm" style (28 fractional hex digits) or subnormal/inf/nan.
    inline int format_ieee_binary128_hexa(char* dst, size_t dst_sz, uint64_t hi, uint64_t lo)
    {
        const unsigned sign = static_cast<unsigned>(hi >> 63);
        const unsigned exp_field = static_cast<unsigned>((hi >> 48) & 0x7FFFu);
        const uint64_t frac_hi = hi & ((((uint64_t)1) << 48) - 1u);
        const uint64_t frac_lo = lo;
        const char* const sgn = sign ? "-" : "+";

        if (exp_field == 0x7FFF)
        {
            if (frac_hi == 0 && frac_lo == 0)
                return std::snprintf(dst, dst_sz, "%sinf", sgn);
            char payload[32];
            for (int j = 0; j < 28; ++j)
            {
                const unsigned nib = ieee112_fraction_nibble(frac_hi, frac_lo, j);
                payload[j] = static_cast<char>(nib < 10 ? ('0' + nib) : ('a' + (nib - 10)));
            }
            payload[28] = '\0';
            return std::snprintf(dst, dst_sz, "%snan(0x%s)", sgn, payload);
        }
        if (exp_field == 0 && frac_hi == 0 && frac_lo == 0)
            return std::snprintf(dst, dst_sz, "%s0x0.0p+0", sgn);

        char mant[32];
        for (int j = 0; j < 28; ++j)
        {
            const unsigned nib = ieee112_fraction_nibble(frac_hi, frac_lo, j);
            mant[j] = static_cast<char>(nib < 10 ? ('0' + nib) : ('a' + (nib - 10)));
        }
        mant[28] = '\0';

        if (exp_field == 0)
        {
            const int exp_unbias = 1 - 16383; // -16382, subnormal
            return std::snprintf(dst, dst_sz, "%s0x0.%sp%+d", sgn, mant, exp_unbias);
        }
        const int exp_unbias = static_cast<int>(exp_field) - 16383;
        return std::snprintf(dst, dst_sz, "%s0x1.%sp%+d", sgn, mant, exp_unbias);
    }
} // namespace ts

// ============================================================================
// Helper: Print a complete value line with all formats (single line, fixed width)
// Format: (dec_hi, dec_lo) -> (hex_hi, hex_lo) -> dec_full -> hex_full
// Uses appropriate precision: 6/7 digits for float, 13/15 for double
// ============================================================================
inline void print_value_line(const char* label, const fpmp_type& mp_val, const fprf_type& ref_val)
{
    double hi = static_cast<double>(mp_val.hi());
    double lo = static_cast<double>(mp_val.lo());

    // Don't show signed zero differences in logs
    if (hi == 0.0) hi = 0.0;
    if (lo == 0.0) lo = 0.0;
    
    // Choose precision based on component type
    // Note: We use separate fprintf for each field to ensure proper alignment
    // since the %e format doesn't guarantee fixed exponent width across values
    using ComponentType = mp_component_t<fpmp_type>;
    
    // Format decimal value with fixed-width output: sign + mantissa + e + sign + 3-digit exponent
    // For float (precision 7):  +X.XXXXXXXe+XXX = 14 chars
    // For double (precision 16): +X.XXXXXXXXXXXXXXXXe+XXX = 23 chars
    auto format_dec = [](double val, int precision, int total_width, char* buf) {
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%+.*e", precision, val);
        
        char* e_pos = strchr(tmp, 'e');
        if (e_pos != nullptr)
        {
            int exp_val = atoi(e_pos + 1);
            *e_pos = '\0';
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
            snprintf(buf, 64, "%*se%+04d", total_width - 5, tmp, exp_val);
#pragma GCC diagnostic pop
        }
        else
        {
            snprintf(buf, 64, "%*s", total_width, tmp);
        }
    };
    
    // Format hex value with fixed-width output: sign + 0x + mantissa + p + sign + 4-digit exponent
    // For float (precision 6):  +0x1.XXXXXXp+XXXX = 18 chars
    // For double (precision 13): +0x1.XXXXXXXXXXXXXp+XXXX = 25 chars
    auto format_hex = [](double val, int precision, int total_width, char* buf) {
        char tmp[48];
        snprintf(tmp, sizeof(tmp), "%+.*a", precision, val);
        
        char* p_pos = strchr(tmp, 'p');
        if (p_pos != nullptr)
        {
            int exp_val = atoi(p_pos + 1);
            *p_pos = '\0';
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
            snprintf(buf, 64, "%*sp%+05d", total_width - 6, tmp, exp_val);  // -6 for "p+XXXX"
#pragma GCC diagnostic pop
        }
        else
        {
            snprintf(buf, 64, "%*s", total_width, tmp);
        }
    };
    
    char hi_dec[64], lo_dec[64], full_dec[128];
    char hi_hex[64], lo_hex[64], full_hex[256];
    
    if constexpr (std::is_same_v<ComponentType, float>)
    {
        double full = static_cast<double>(ref_val);
        if (full == 0.0) full = 0.0;
        // fp32mp2: 7 decimal digits (14 chars), 6 hex digits (18 chars)
        // Full value uses double precision (sufficient for 46-bit mantissa)
        format_dec(hi, 7, 14, hi_dec);
        format_dec(lo, 7, 14, lo_dec);
        format_dec(full, 16, 23, full_dec);
        format_hex(hi, 6, 18, hi_hex);
        format_hex(lo, 6, 18, lo_hex);
        format_hex(full, 13, 25, full_hex);
        fprintf(stderr, "  %-6s (%s, %s) == (%s, %s) == %s == %s\n",
                label, hi_dec, lo_dec, hi_hex, lo_hex, full_dec, full_hex);
    }
    else
    {
        // fp64mp2: hi/lo columns use double formatting; full column uses exact binary128 hex
        // plus a double-based decimal (not round-trip exact — for a quick magnitude read).
        format_dec(hi, 16, 23, hi_dec);
        format_dec(lo, 16, 23, lo_dec);
        format_hex(hi, 13, 25, hi_hex);
        format_hex(lo, 13, 25, lo_hex);
        
        // Full __ts_fp128: exact IEEE binary128 hex (bit-accurate); decimal via double (skim only).
#if (TS_HAS_LIBQUADMATH == 1) || (TS_HAS_LDOUBLE128 == 1)
        {
            uint64_t w_hi = 0, w_lo = 0;
            ts::read_ieee_binary128_u64_pair(&ref_val, &w_hi, &w_lo);
            char tmp[256];
            ts::format_ieee_binary128_hexa(tmp, sizeof(tmp), w_hi, w_lo);
            char* p_pos = strchr(tmp, 'p');
            if (p_pos != nullptr)
            {
                int exp_val = atoi(p_pos + 1);
                *p_pos = '\0';
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
                snprintf(full_hex, sizeof(full_hex), "%sp%+05d", tmp, exp_val);
#pragma GCC diagnostic pop
            }
            else
            {
                snprintf(full_hex, sizeof(full_hex), "%s", tmp);
            }
        }
        {
            char tmp[64];
            double d_approx = static_cast<double>(ref_val);
            if (d_approx == 0.0)
                d_approx = 0.0;
            snprintf(tmp, sizeof(tmp), "%+.16e", d_approx);
            char* e_pos = strchr(tmp, 'e');
            if (e_pos == nullptr)
                e_pos = strchr(tmp, 'E');
            if (e_pos != nullptr)
            {
                int exp_val = atoi(e_pos + 1);
                *e_pos = '\0';
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
                snprintf(full_dec, sizeof(full_dec), "%se%+04d", tmp, exp_val);
#pragma GCC diagnostic pop
            }
            else
            {
                snprintf(full_dec, sizeof(full_dec), "%s", tmp);
            }
        }
#else
        #error "fp64mp2 print_value_line requires TS_HAS_LIBQUADMATH or TS_HAS_LDOUBLE128"
#endif

        fprintf(stderr, "  %-6s (%s, %s) == (%s, %s) == %s == %s\n",
                label, hi_dec, lo_dec, hi_hex, lo_hex, full_dec, full_hex);
    }
}

// ============================================================================
// Helper: Print binary representation of a multi-precision value
// ============================================================================
inline void print_binary_mp(const char* label, const fpmp_type& val)
{
    using ComponentType = mp_component_t<fpmp_type>;
    if constexpr (std::is_same_v<ComponentType, float>)
    {
        // Ignore sign of zero for binary dumps
        float hi_f = val.hi();
        float lo_f = val.lo();
        if (hi_f == 0.0f) hi_f = 0.0f;
        if (lo_f == 0.0f) lo_f = 0.0f;

        uint32_t hi_bits = ts::bit_cast<uint32_t>(hi_f);
        uint32_t lo_bits = ts::bit_cast<uint32_t>(lo_f);
        uint64_t combined = (static_cast<uint64_t>(hi_bits) << 32) | lo_bits;
        fprintf(stderr, "  %-6s %s\n", label, ts::ll2bin(combined).c_str());
    }
    else
    {
        // Ignore sign of zero for binary dumps
        double hi_d = val.hi();
        double lo_d = val.lo();
        if (hi_d == 0.0) hi_d = 0.0;
        if (lo_d == 0.0) lo_d = 0.0;

        uint64_t hi_bits = ts::bit_cast<uint64_t>(hi_d);
        uint64_t lo_bits = ts::bit_cast<uint64_t>(lo_d);
        ts::uint128_t combined = (static_cast<ts::uint128_t>(hi_bits) << 64) | lo_bits;
        fprintf(stderr, "  %-6s %s\n", label, ts::ll2bin(combined).c_str());
    }
}

// ============================================================================
// Helper: Print a signed integer value line (for int2mp, ll2mp, mp2int, mp2ll)
// Uses raw bit extraction to preserve all bits exactly (see to_fpmp_for_storage)
// ============================================================================
inline void print_int_value_line(const char* label, const fpmp_type& mp_val, bool is_64bit = true)
{
    using ComponentType = mp_component_t<fpmp_type>;
    
    if (is_64bit)
    {
        uint64_t bits;
        if constexpr (std::is_same_v<ComponentType, float>) {
            // fp32mp2: hi stores upper 32 bits, lo stores lower 32 bits (as raw floats)
            float f_hi = mp_val.hi();
            float f_lo = mp_val.lo();
            uint32_t hi32, lo32;
            memcpy(&hi32, &f_hi, sizeof(uint32_t));
            memcpy(&lo32, &f_lo, sizeof(uint32_t));
            bits = (static_cast<uint64_t>(hi32) << 32) | static_cast<uint64_t>(lo32);
        } else {
            // fp64mp2: hi stores all 64 bits as raw double
            double d = mp_val.hi();
            memcpy(&bits, &d, sizeof(uint64_t));
        }
        int64_t int_val = static_cast<int64_t>(bits);
        fprintf(stderr, "  %-6s %+21lld (0x%016llx)\n", label, (long long)int_val, (unsigned long long)bits);
    }
    else
    {
        // For 32-bit integers
        if constexpr (std::is_same_v<ComponentType, float>) {
            // fp32mp2: value is stored as raw bits in hi component
            float f = mp_val.hi();
            uint32_t bits;
            memcpy(&bits, &f, sizeof(uint32_t));
            int32_t int_val = static_cast<int32_t>(bits);
            fprintf(stderr, "  %-6s %+21d (0x%08x)\n", label, int_val, bits);
        } else {
            // fp64mp2: value is directly in hi() as a double
            int32_t int_val = static_cast<int32_t>(mp_val.hi());
            fprintf(stderr, "  %-6s %+21d (0x%08x)\n", label, int_val, static_cast<uint32_t>(int_val));
        }
    }
}

// ============================================================================
// Helper: Print an unsigned integer value line (for uint2mp, ull2mp, mp2uint, mp2ull)
// Uses raw bit extraction to preserve all bits exactly (see to_fpmp_for_storage)
// ============================================================================
inline void print_uint_value_line(const char* label, const fpmp_type& mp_val, bool is_64bit = true)
{
    using ComponentType = mp_component_t<fpmp_type>;
    
    if (is_64bit)
    {
        uint64_t uint_val;
        if constexpr (std::is_same_v<ComponentType, float>) {
            // fp32mp2: hi stores upper 32 bits, lo stores lower 32 bits (as raw floats)
            float f_hi = mp_val.hi();
            float f_lo = mp_val.lo();
            uint32_t hi32, lo32;
            memcpy(&hi32, &f_hi, sizeof(uint32_t));
            memcpy(&lo32, &f_lo, sizeof(uint32_t));
            uint_val = (static_cast<uint64_t>(hi32) << 32) | static_cast<uint64_t>(lo32);
        } else {
            // fp64mp2: hi stores all 64 bits as raw double
            double d = mp_val.hi();
            memcpy(&uint_val, &d, sizeof(uint64_t));
        }
        fprintf(stderr, "  %-6s %21llu (0x%016llx)\n", label, (unsigned long long)uint_val, (unsigned long long)uint_val);
    }
    else
    {
        // For 32-bit integers
        if constexpr (std::is_same_v<ComponentType, float>) {
            // fp32mp2: value is stored as raw bits in hi component
            float f = mp_val.hi();
            uint32_t uint_val;
            memcpy(&uint_val, &f, sizeof(uint32_t));
            fprintf(stderr, "  %-6s %21u (0x%08x)\n", label, uint_val, uint_val);
        } else {
            // fp64mp2: value is directly in hi() as a double
            uint32_t uint_val = static_cast<uint32_t>(mp_val.hi());
            fprintf(stderr, "  %-6s %21u (0x%08x)\n", label, uint_val, uint_val);
        }
    }
}

// ============================================================================
// Helper: Print a boolean/comparison result line
// ============================================================================
inline void print_bool_value_line(const char* label, const fpmp_type& mp_val)
{
    // For comparison results, the value is 0 or 1 (false/true)
    int int_val = static_cast<int>(mp_val.hi());
    const char* bool_str = (int_val != 0) ? "true" : "false";
    fprintf(stderr, "  %-6s %d (%s)\n", label, int_val, bool_str);
}

// ============================================================================
// Helper: Print a double (floating-point) value line (for fp2mp input)
// Reconstructs the original double from mp storage and prints as dec + hex
// ============================================================================
inline void print_double_value_line(const char* label, const fpmp_type& mp_val)
{
    double val = static_cast<double>(mp_val.hi()) + static_cast<double>(mp_val.lo());
    if (val == 0.0) val = 0.0;
    fprintf(stderr, "  %-6s %+.16e == %+.13a\n", label, val, val);
}

// ============================================================================
// Helper: Detect if function name indicates special input arguments or results
// Returns: 0 = no special handling
//          1 = arg0 is signed 32-bit integer (int2mp)
//          2 = result is signed 32-bit integer (mp2int)
//          3 = result is bool (eq, ne, lt, le, gt, ge)
//          4 = arg0 is unsigned 32-bit integer (uint2mp)
//          5 = result is unsigned 32-bit integer (mp2uint)
//          6 = result is unsigned 64-bit integer (mp2ull)
//          7 = input is signed 64-bit integer (ll2mp)
//          8 = input is unsigned 64-bit integer (ull2mp)
//          9 = result is signed 64-bit integer (mp2ll)
//         10 = arg0 is floating-point / double (fp2mp)
// ============================================================================
inline int get_integer_arg_type(const char* func_name)
{
    if (func_name == nullptr) return 0;
    // Floating-point (double) input: fp2mp
    if (strstr(func_name, "fp2mp"))
        return 10;
    // 64-bit unsigned integer input: ull2mp
    if (strstr(func_name, "ull2mp"))
        return 8;
    // 32-bit unsigned integer input: uint2mp
    if (strstr(func_name, "uint2mp"))
        return 4;
    // 64-bit signed integer input: ll2mp
    if (strstr(func_name, "ll2mp"))
        return 7;
    // 32-bit signed integer input: int2mp
    if (strstr(func_name, "int2mp"))
        return 1;
    // 64-bit unsigned integer result: mp2ull
    if (strstr(func_name, "mp2ull"))
        return 6;
    // 32-bit unsigned integer result: mp2uint
    if (strstr(func_name, "mp2uint"))
        return 5;
    // 64-bit signed integer result: mp2ll (check before mp2int since "mp2ll" doesn't contain "mp2int")
    if (strstr(func_name, "mp2ll"))
        return 9;
    // 32-bit signed integer result: mp2int
    if (strstr(func_name, "mp2int"))
        return 2;
    // Comparison operators: result is bool (eq, ne, lt, le, gt, ge)
    // Match function names at the start (e.g., "eq <fast>")
    if (strncmp(func_name, "eq ", 3) == 0 || strncmp(func_name, "ne ", 3) == 0 ||
        strncmp(func_name, "lt ", 3) == 0 || strncmp(func_name, "le ", 3) == 0 ||
        strncmp(func_name, "gt ", 3) == 0 || strncmp(func_name, "ge ", 3) == 0)
        return 3;
    return 0;
}

// ============================================================================
// Helper: Print a single error record with all formats
// ============================================================================
inline void print_error_record(const accuracy_error_record_t& e, const char* label, int max_bits, 
                               const char* func_name = nullptr)
{
    int correct_bits = compute_correct_bits(e.rel_err, max_bits);
    
    // Get classification string
    const char* class_str = (e.classification == record_class::error)   ? "error" :
                            (e.classification == record_class::warning) ? "warning" : "normal";

    bool is_mp2fp = func_name && strncmp(func_name, "mp2fp", 5) == 0;
    
    fprintf(stderr, "--- %s --- rel_err: %.4e, bits: %d/%d = %s\n", label, e.rel_err, correct_bits, max_bits, class_str);
    
    // Detect if this function has integer arguments or results
    // Types: 1=int32 input, 4=uint32 input, 7=int64 input, 8=uint64 input
    //        2=int32/int64 result, 5=uint32 result, 6=uint64 result, 3=bool result
    int int_type = get_integer_arg_type(func_name);
    
    // Print input arguments (fpmp_type -> fprf_type)
    for (int i = 0; i < e.arity; i++)
    {
        char arg_label[16];
        snprintf(arg_label, sizeof(arg_label), "arg%d:", i);
        
        // For int2mp-like functions, arg0 is an integer
        if (int_type == 1 && i == 0)
        {
            // Signed 32-bit integer input (int2mp)
            print_int_value_line(arg_label, e.args[i], false);
        }
        else if (int_type == 7 && i == 0)
        {
            // Signed 64-bit integer input (ll2mp)
            print_int_value_line(arg_label, e.args[i], true);
        }
        else if (int_type == 4 && i == 0)
        {
            // Unsigned 32-bit integer input (uint2mp)
            print_uint_value_line(arg_label, e.args[i], false);
        }
        else if (int_type == 8 && i == 0)
        {
            // Unsigned 64-bit integer input (ull2mp)
            print_uint_value_line(arg_label, e.args[i], true);
        }
        else if (int_type == 10 && i == 0)
        {
            // Floating-point (double) input (fp2mp)
            print_double_value_line(arg_label, e.args[i]);
        }
        else
        {
            fprf_type arg_as_ref = static_cast<fprf_type>(e.args[i]);
            print_value_line(arg_label, e.args[i], arg_as_ref);
        }
    }
    
    // Print test result
    if (is_mp2fp)
    {
        print_double_value_line("Test:", e.test_result);
    }
    else if (int_type == 2)
    {
        // For mp2int: result is signed 32-bit integer
        print_int_value_line("Test:", e.test_result, false);
    }
    else if (int_type == 9)
    {
        // For mp2ll: result is signed 64-bit integer
        print_int_value_line("Test:", e.test_result, true);
    }
    else if (int_type == 5)
    {
        // For mp2uint: result is unsigned 32-bit integer
        print_uint_value_line("Test:", e.test_result, false);
    }
    else if (int_type == 6)
    {
        // For mp2ull: result is unsigned 64-bit integer
        print_uint_value_line("Test:", e.test_result, true);
    }
    else if (int_type == 3)
    {
        // For comparison functions, result is a bool
        print_bool_value_line("Test:", e.test_result);
    }
    else
    {
        fprf_type test_as_ref = static_cast<fprf_type>(e.test_result);
        print_value_line("Test:", e.test_result, test_as_ref);
    }

    // Print reference result
    if (is_mp2fp)
    {
        fpmp_type ref_as_mp = ts_ref_to_fpmp<fpmp_type>(e.ref_result);
        print_double_value_line("Ref:", ref_as_mp);
        print_binary_mp("TstBin:", e.test_result);
        print_binary_mp("RefBin:", ref_as_mp);
    }
    else if (int_type == 2)
    {
        // Signed 32-bit integer result (mp2int)
        int32_t ref_int = static_cast<int32_t>(e.ref_result);
        fprintf(stderr, "  %-6s %+21d (0x%08x)\n", "Ref:", ref_int, static_cast<uint32_t>(ref_int));
    }
    else if (int_type == 9)
    {
        // Signed 64-bit integer result (mp2ll)
        int64_t ref_int = static_cast<int64_t>(e.ref_result);
        fprintf(stderr, "  %-6s %+21lld (0x%016llx)\n", "Ref:", (long long)ref_int, (unsigned long long)ref_int);
    }
    else if (int_type == 5)
    {
        // Unsigned 32-bit integer result (mp2uint)
        uint32_t ref_uint = static_cast<uint32_t>(e.ref_result);
        fprintf(stderr, "  %-6s %21u (0x%08x)\n", "Ref:", ref_uint, ref_uint);
    }
    else if (int_type == 6)
    {
        // Unsigned 64-bit integer result (mp2ull)
        uint64_t ref_ull = static_cast<uint64_t>(e.ref_result);
        fprintf(stderr, "  %-6s %21llu (0x%016llx)\n", "Ref:", (unsigned long long)ref_ull, (unsigned long long)ref_ull);
    }
    else if (int_type == 3)
    {
        // Boolean result (eq, ne, lt, le, gt, ge)
        int ref_bool = static_cast<int>(e.ref_result);
        const char* bool_str = (ref_bool != 0) ? "true" : "false";
        fprintf(stderr, "  %-6s %d (%s)\n", "Ref:", ref_bool, bool_str);
    }
    else
    {
        fpmp_type ref_as_mp = ts_ref_to_fpmp<fpmp_type>(e.ref_result);
        print_value_line("Ref:", ref_as_mp, e.ref_result);
        
        // Print binary representations for test and reference (only for floating-point results)
        print_binary_mp("TstBin:", e.test_result);
        print_binary_mp("RefBin:", ref_as_mp);
    }

    fprintf(stderr,"\n");
}

// ============================================================================
// Print accuracy error log (to stderr)
// ============================================================================
inline void print_accuracy_log(const ts_accuracy_result_t& result, 
                               const char* func_name = nullptr,
                               const char* dataset_name = nullptr,
                               uint64_t max_entries = TS_ACCURACY_ERROR_LOG_SIZE)
{
    uint64_t logged_count = std::min(result.error_log_count, (uint64_t)TS_ACCURACY_ERROR_LOG_SIZE);
    uint64_t print_count  = std::min(logged_count, max_entries);
    
    // Header
    fprintf(stderr, "\n");
    print_separator(stderr, '=');
    if (func_name && dataset_name)
        fprintf(stderr, "Accuracy Error Log: %s [%s]\n", func_name, dataset_name);
    else if (func_name)
        fprintf(stderr, "Accuracy Error Log: %s\n", func_name);
    else if (dataset_name)
        fprintf(stderr, "Accuracy Error Log: [%s]\n", dataset_name);
    else
        fprintf(stderr, "Accuracy Error Log\n");
    print_separator(stderr, '=');
    fprintf(stderr, "Total errors detected:   %lu (rel_err > error_threshold)\n", (unsigned long)result.total_errs);
    fprintf(stderr, "Total warnings detected: %lu (rel_err > warning_threshold)\n", (unsigned long)result.total_warnings);
    fprintf(stderr, "Errors logged:           %lu (buffer size: %d)\n", 
            (unsigned long)logged_count, TS_ACCURACY_ERROR_LOG_SIZE);
    if (print_count < logged_count)
        fprintf(stderr, "Showing first:           %lu entries\n", (unsigned long)print_count);
    print_separator(stderr);
    
    int max_bits = result.max_mantissa_bits;
    constexpr bool is_mp = is_multiprecision_v<fpmp_type>;
    constexpr double error_threshold   = get_error_threshold<mp_component_t<fpmp_type>>();
    constexpr double warning_threshold = get_warning_threshold<mp_component_t<fpmp_type>>();
    fprintf(stderr, "Test type:         %s\n", is_mp ? "multi-precision (hi, lo)" : "scalar");
    fprintf(stderr, "Error threshold:   %.2e\n", error_threshold);
    fprintf(stderr, "Warning threshold: %.2e\n", warning_threshold);
    print_separator(stderr);
    
    // Always print the maximum error (tracked regardless of threshold)
    if (result.has_max_error)
    {
        print_error_record(result.max_error_record, "MAX", max_bits, func_name);
        print_separator(stderr);
    }
    else
    {
        fprintf(stderr, "  (no valid comparisons made)\n");
        fprintf(stderr, "====================================================================================\n\n");
        return;
    }
    
    // Print additional error records (above threshold only)
    if (print_count == 0)
    {
        fprintf(stderr, "  (no additional issues detected above threshold)\n");
    }
    else
    {
        uint64_t printed = 0;
        for (uint64_t i = 0; i < logged_count && printed < print_count; i++)
        {
            // Skip if this is the same as max (compare by rel_err)
            if (result.error_log[i].rel_err == result.max_error_record.rel_err) continue;
            
            char label[24];
            snprintf(label, sizeof(label), "%lu", (unsigned long)printed);
            print_error_record(result.error_log[i], label, max_bits, func_name);
            printed++;
        }
    }
    
    print_separator(stderr, '=');
    if (result.error_log_count > logged_count)
    {
        fprintf(stderr, "Note: %lu additional errors occurred but were not logged (buffer full)\n",
                (unsigned long)(result.error_log_count - logged_count));
    }
    fprintf(stderr, "\n");
}

// Helper: format percentage - use scientific notation for very small values, always with %
inline void format_percentage(char* buf, size_t buf_size, double pct)
{
    if (pct >= 0.01)
    {
        snprintf(buf, buf_size, "%6.2f%%", pct);
    }
    else if (pct > 0.0)
    {
        snprintf(buf, buf_size, "%5.0e%%", pct);  // e.g. "3e-05%"
    }
    else
    {
        snprintf(buf, buf_size, "%6.2f%%", 0.0);
    }
}

// ============================================================================
// Print per-class accuracy statistics summary
// ============================================================================
inline void print_class_statistics(const ts_accuracy_result_t& result,
                                   const char* func_name = nullptr,
                                   const char* dataset_name = nullptr,
                                   FILE* out = stderr)
{
    // Header
    fprintf(out, "\n");
    print_separator(out, '=');
    if (func_name && dataset_name)
        fprintf(out, "Accuracy Classification Summary: %s [%s]\n", func_name, dataset_name);
    else if (func_name)
        fprintf(out, "Accuracy Classification Summary: %s\n", func_name);
    else if (dataset_name)
        fprintf(out, "Accuracy Classification Summary: [%s]\n", dataset_name);
    else
        fprintf(out, "Accuracy Classification Summary\n");
    print_separator(out, '=');
    
    // Calculate total samples from all classes
    uint64_t total_samples = 0;
    for (int c = 0; c < ACCURACY_CLASS_COUNT; c++)
    {
        total_samples += result.class_results[c].count;
    }
    
    // Table header (added Percent column)
    fprintf(out, "%-20s | %12s | %8s | %12s | %12s | %5s\n",
            "Class", "Count", "Percent", "Max RelErr", "Avg RelErr", "Bits");
    fprintf(out, "---------------------+--------------+----------+--------------+--------------+------\n");
    
    // Count non-empty classes for summary
    int non_empty_warnings = 0;
    int non_empty_errors = 0;
    bool has_any_rows = false;
    
    // Print each class that has any entries
    for (int c = 0; c < ACCURACY_CLASS_COUNT; c++)
    {
        const accuracy_class_result_t& cr = result.class_results[c];
        if (cr.count == 0) continue;
        
        has_any_rows = true;
        accuracy_class cls = static_cast<accuracy_class>(c);
        const char* name = accuracy_class_name(cls);
        
        if (cls == accuracy_class::error)
            non_empty_errors++;
        else if (is_warning_class(cls))
            non_empty_warnings++;
        
        // Calculate percentage
        double pct = (total_samples > 0) ? (cr.count * 100.0 / total_samples) : 0.0;
        char pct_buf[16];
        format_percentage(pct_buf, sizeof(pct_buf), pct);
        
        // For special classes (input_special, output_special), no rel_err is computed
        // Show "--" instead of 0.00e+00
        bool has_rel_err = (cls != accuracy_class::input_special && 
                            cls != accuracy_class::output_special);
        
        if (has_rel_err)
        {
            if (cr.correct_bits >= 0)
            {
                fprintf(out, "%-20s | %12lu | %8s | %12.2e | %12.2e | %5d\n",
                        name,
                        (unsigned long)cr.count,
                        pct_buf,
                        cr.max_rel_err,
                        cr.avg_rel_err,
                        cr.correct_bits);
            }
            else
            {
                fprintf(out, "%-20s | %12lu | %8s | %12.2e | %12.2e | %5s\n",
                        name,
                        (unsigned long)cr.count,
                        pct_buf,
                        cr.max_rel_err,
                        cr.avg_rel_err,
                        "N/A");
            }
        }
        else
        {
            // Special class without rel_err
            fprintf(out, "%-20s | %12lu | %8s | %12s | %12s | %5s\n",
                    name,
                    (unsigned long)cr.count,
                    pct_buf,
                    "--",
                    "--",
                    "--");
        }
    }
    
    // If no classes had entries, show "normal (OK)" with 100%
    if (!has_any_rows)
    {
        fprintf(out, "%-20s | %12lu | %7.2f%% | %12s | %12s | %5s\n",
                "normal (OK)", (unsigned long)total_samples, 100.0, "--", "--", "--");
    }
    
    // Print total row
    fprintf(out, "---------------------+--------------+----------+--------------+--------------+------\n");
    fprintf(out, "%-20s | %12lu | %7.2f%% |\n", "TOTAL", (unsigned long)total_samples, 100.0);
    fprintf(out, "---------------------+--------------+----------+--------------+--------------+------\n");
    fprintf(out, "Warning class count:        %d\n", non_empty_warnings);
    fprintf(out, "Error class count:          %lu\n", (unsigned long)result.total_errs);
    fprintf(out, "====================================================================================\n\n");
}

// ============================================================================
// Print per-class max error records (detailed view)
// ============================================================================
inline void print_class_max_errors(const ts_accuracy_result_t& result,
                                   const char* func_name = nullptr,
                                   const char* dataset_name = nullptr)
{
    int max_bits = result.max_mantissa_bits;
    
    fprintf(stderr, "\n");
    print_separator(stderr, '=');
    if (func_name && dataset_name)
        fprintf(stderr, "Per-Class Maximum Errors: %s [%s]\n", func_name, dataset_name);
    else if (func_name)
        fprintf(stderr, "Per-Class Maximum Errors: %s\n", func_name);
    else
        fprintf(stderr, "Per-Class Maximum Errors\n");
    print_separator(stderr, '=');
    
    bool has_any = false;
    for (int c = 0; c < ACCURACY_CLASS_COUNT; c++)
    {
        const accuracy_class_result_t& cr = result.class_results[c];
        if (!cr.has_max_record || cr.count == 0) continue;
        
        has_any = true;
        accuracy_class cls = static_cast<accuracy_class>(c);
        char label[32];
        snprintf(label, sizeof(label), "[%s]", accuracy_class_short_name(cls));
        
        fprintf(stderr, "\n--- %s (count: %lu) ---\n", 
                accuracy_class_name(cls), (unsigned long)cr.count);
        print_error_record(cr.max_record, label, max_bits, func_name);
    }
    
    if (!has_any)
    {
        fprintf(stderr, "  (no per-class errors recorded)\n");
    }
    
    fprintf(stderr, "====================================================================================\n\n");
}

// ============================================================================
// Special Values Table - Format a result value for table display (compact)
// ============================================================================
template<typename T>
inline const char* format_special_result(T value, char* buf, size_t buf_size)
{
    if constexpr (std::is_floating_point_v<T> || is_multiprecision_v<T>)
    {
        double d;
        if constexpr (is_multiprecision_v<T>)
            d = static_cast<double>(value.hi());  // Use hi component for display
        else
            d = static_cast<double>(value);
        
        if (std::isnan(d)) {
            snprintf(buf, buf_size, "nan");
        } else if (std::isinf(d)) {
            snprintf(buf, buf_size, "%cinf", d < 0 ? '-' : '+');
        } else if (d == 0.0) {
            // Check sign of zero
            uint64_t bits;
            memcpy(&bits, &d, sizeof(bits));
            snprintf(buf, buf_size, "%c0", (bits >> 63) ? '-' : '+');
        } else if (std::fabs(d) >= 1e6 || std::fabs(d) < 1e-3) {
            snprintf(buf, buf_size, "%.1e", d);  // 1 decimal digit for scientific
        } else if (std::fabs(d) >= 10.0) {
            snprintf(buf, buf_size, "%.0f", d);  // No decimals for large integers
        } else {
            snprintf(buf, buf_size, "%.2g", d);  // 2 significant digits
        }
    }
    else if constexpr (std::is_integral_v<T>)
    {
        if constexpr (std::is_signed_v<T>)
            snprintf(buf, buf_size, "%lld", (long long)value);
        else
            snprintf(buf, buf_size, "%llu", (unsigned long long)value);
    }
    else
    {
        snprintf(buf, buf_size, "?");
    }
    return buf;
}

// ============================================================================
// Helper: Generate special value for a given type (handles multi-precision)
// ============================================================================
template<typename TestType>
inline TestType generate_special_value_for_type(int index)
{
    if constexpr (is_multiprecision_v<TestType>)
    {
        // For multi-precision types, use the component type for special values
        using ComponentType = mp_component_t<TestType>;
        ComponentType special_val = generate_special_fp_scalar<ComponentType>(index);
        return TestType(special_val, ComponentType(0));  // Put special value in hi, zero in lo
    }
    else
    {
        // For scalar types, generate directly
        return generate_special_fp_scalar<TestType>(index);
    }
}

// ============================================================================
// Print legend for special values table
// ============================================================================
inline void print_special_values_legend(FILE* out, bool is_fp64, int width = 84)
{
    print_separator(out, '-', width);
    fprintf(out, "Legend:\n");
    print_separator(out, '-', width);
    if (is_fp64) {
        fprintf(out, "  -maxN = -1.8e+308 (max norm)    -minN = -2.2e-308 (min norm)\n");
        fprintf(out, "  -maxD = -2.2e-308 (max denorm)  -minD = -4.9e-324 (min denorm)\n");
        fprintf(out, "  +minD = +4.9e-324 (min denorm)  +maxD = +2.2e-308 (max denorm)\n");
        fprintf(out, "  +minN = +2.2e-308 (min norm)    +maxN = +1.8e+308 (max norm)\n");
    } else {
        fprintf(out, "  -maxN = -3.4e+38 (max norm)     -minN = -1.2e-38 (min norm)\n");
        fprintf(out, "  -maxD = -1.2e-38 (max denorm)   -minD = -1.4e-45 (min denorm)\n");
        fprintf(out, "  +minD = +1.4e-45 (min denorm)   +maxD = +1.2e-38 (max denorm)\n");
        fprintf(out, "  +minN = +1.2e-38 (min norm)     +maxN = +3.4e+38 (max norm)\n");
    }
    print_separator(out, '-', width);
    fprintf(out, "\n");
}

// ============================================================================
// Print 2D Special Values Table for binary functions (add, sub, mul, div, etc.)
// ============================================================================
template<typename FuncTag, typename TestType>
inline void print_special_values_table_2d(FILE* out, const char* func_name, 
                                          const char* method_name = nullptr,
                                          const char* dataset_name = nullptr)
{
    using ResultType = tag_result_t<FuncTag>;
    constexpr int N = SPECIAL_TABLE_COUNT;  // Excludes sNaN
    
    // Column width for cells: 8 for fp32 (-3.4e+38), 9 for fp64 (-1.8e+308)
    constexpr bool is_fp64 = is_multiprecision_v<TestType> 
        ? std::is_same_v<mp_component_t<TestType>, double>
        : std::is_same_v<TestType, double>;
    constexpr int CELL_WIDTH = is_fp64 ? 9 : 8;
    constexpr int HEADER_WIDTH = 5;  // Row header width for special value names
    constexpr int TABLE_WIDTH = (HEADER_WIDTH + 1) + N * (CELL_WIDTH + 1) + 1;  // Total table width
    
    fprintf(out, "\n");
    print_separator(out, '=', TABLE_WIDTH);
    if (method_name && dataset_name)
        fprintf(out, "Special Values Table: %s <%s> [%s]\n", func_name, method_name, dataset_name);
    else
        fprintf(out, "Special Values Table: %s\n", func_name);
    print_separator(out, '=', TABLE_WIDTH);
    
    // Print header row with column names (a\b indicates rows=arg1, cols=arg2)
    fprintf(out, "%*s|", HEADER_WIDTH + 1, "a\\b");
    for (int col = 0; col < N; col++) {
        int idx = get_special_table_index(col);
        fprintf(out, " %*s", CELL_WIDTH, get_special_fp_name(idx));
    }
    fprintf(out, "\n");
    
    // Print separator (same width as table)
    print_separator(out, '-', TABLE_WIDTH);
    
    // Print each row
    char buf[32];
    for (int row = 0; row < N; row++) {
        int row_idx = get_special_table_index(row);
        // Row header (first argument)
        fprintf(out, "%*s|", HEADER_WIDTH + 1, get_special_fp_name(row_idx));
        
        TestType arg1 = generate_special_value_for_type<TestType>(row_idx);
        
        for (int col = 0; col < N; col++) {
            int col_idx = get_special_table_index(col);
            TestType arg2 = generate_special_value_for_type<TestType>(col_idx);
            
            // Compute result using the function tag
            ResultType result = FuncTag{}(arg1, arg2);
            
            format_special_result(result, buf, sizeof(buf));
            fprintf(out, " %*s", CELL_WIDTH, buf);
        }
        fprintf(out, "\n");
    }
    
    print_special_values_legend(out, is_fp64, TABLE_WIDTH);
}

// ============================================================================
// Print 1D Special Values Table for unary functions (sqrt, rsqrt, exp, etc.)
// ============================================================================
template<typename FuncTag, typename TestType>
inline void print_special_values_table_1d(FILE* out, const char* func_name,
                                          const char* method_name = nullptr,
                                          const char* dataset_name = nullptr)
{
    using ResultType = tag_result_t<FuncTag>;
    constexpr int N = SPECIAL_TABLE_COUNT;  // Excludes sNaN
    
    // Detect if fp64 for legend
    constexpr bool is_fp64 = is_multiprecision_v<TestType> 
        ? std::is_same_v<mp_component_t<TestType>, double>
        : std::is_same_v<TestType, double>;
    
    constexpr int TABLE_WIDTH = 84;
    
    fprintf(out, "\n");
    print_separator(out, '=', TABLE_WIDTH);
    if (method_name && dataset_name)
        fprintf(out, "Special Values Table: %s <%s> [%s]\n", func_name, method_name, dataset_name);
    else
        fprintf(out, "Special Values Table: %s\n", func_name);
    print_separator(out, '=', TABLE_WIDTH);
    fprintf(out, "%-5s | %-10s | %-10s\n", "Input", "Value", "Result");
    print_separator(out, '-', TABLE_WIDTH);
    
    char val_buf[32], res_buf[32];
    for (int i = 0; i < N; i++) {
        int idx = get_special_table_index(i);
        TestType arg = generate_special_value_for_type<TestType>(idx);
        ResultType result = FuncTag{}(arg);
        
        format_special_result(arg, val_buf, sizeof(val_buf));
        format_special_result(result, res_buf, sizeof(res_buf));
        
        fprintf(out, "%-5s | %10s | %10s\n", 
                get_special_fp_name(idx), val_buf, res_buf);
    }
    
    print_special_values_legend(out, is_fp64, TABLE_WIDTH);
}

// ============================================================================
// Print 1D Special Values Table for integer output functions (mp2int, mp2ll, etc.)
// Uses different special values focused on integer boundaries
// ============================================================================
template<typename FuncTag, typename TestType>
inline void print_special_values_table_int_output(FILE* out, const char* func_name,
                                                   const char* method_name = nullptr,
                                                   const char* dataset_name = nullptr)
{
    using ResultType = tag_result_t<FuncTag>;
    constexpr int N = SPECIAL_INT_OUTPUT_COUNT;
    
    // Detect component type for value generation
    using ComponentType = std::conditional_t<is_multiprecision_v<TestType>,
                                             mp_component_t<TestType>, TestType>;
    
    constexpr int TABLE_WIDTH = 84;
    
    fprintf(out, "\n");
    print_separator(out, '=', TABLE_WIDTH);
    if (method_name && dataset_name)
        fprintf(out, "Special Values Table: %s <%s> [%s] (integer output)\n", func_name, method_name, dataset_name);
    else
        fprintf(out, "Special Values Table: %s (integer output)\n", func_name);
    print_separator(out, '=', TABLE_WIDTH);
    fprintf(out, "%-6s | %-15s | %-20s\n", "Input", "Value", "Result");
    print_separator(out, '-', TABLE_WIDTH);
    
    char val_buf[32], res_buf[32];
    for (int i = 0; i < N; i++) {
        // Generate special value using component type
        ComponentType special_val = generate_special_int_output_scalar<ComponentType>(i);
        TestType arg;
        if constexpr (is_multiprecision_v<TestType>) {
            arg = TestType(special_val, ComponentType(0));
        } else {
            arg = special_val;
        }
        
        ResultType result = FuncTag{}(arg);
        
        format_special_result(arg, val_buf, sizeof(val_buf));
        format_special_result(result, res_buf, sizeof(res_buf));
        
        fprintf(out, "%-6s | %15s | %20s\n", 
                get_special_int_output_name(i), val_buf, res_buf);
    }
    
    // Legend for integer output table
    print_separator(out, '-', TABLE_WIDTH);
    fprintf(out, "Legend: -LMAX = -2^63 (llong min), -IMAX = -2^31 (int min)\n");
    fprintf(out, "        +IMAX = 2^31-1 (int max), +UMAX = 2^32-1 (uint max), +LMAX = 2^63-1 (llong max)\n");
    print_separator(out, '-', TABLE_WIDTH);
    fprintf(out, "\n");
}

// ============================================================================
// Print 1D Special Values Table for integer input functions (int2mp, ll2mp, etc.)
// Uses integer special values from ts_dataset.hpp
// ============================================================================
template<typename FuncTag, typename TestType>
inline void print_special_values_table_int_input(FILE* out, const char* func_name,
                                                  const char* method_name = nullptr,
                                                  const char* dataset_name = nullptr)
{
    using InputType = typename FuncTag::input_type;
    using ResultType = tag_result_t<FuncTag>;
    constexpr int N = SPECIAL_INT_INPUT_COUNT;  // 16 entries
    
    // Different column widths based on integer type
    constexpr bool is_64bit = std::is_same_v<InputType, int64_t> || std::is_same_v<InputType, uint64_t>;
    constexpr int VALUE_WIDTH = is_64bit ? 22 : 14;
    constexpr int TABLE_WIDTH = 84;
    
    fprintf(out, "\n");
    print_separator(out, '=', TABLE_WIDTH);
    if (method_name && dataset_name)
        fprintf(out, "Special Values Table: %s <%s> [%s] (integer input)\n", func_name, method_name, dataset_name);
    else
        fprintf(out, "Special Values Table: %s (integer input)\n", func_name);
    print_separator(out, '=', TABLE_WIDTH);
    
    fprintf(out, "%-5s | %*s | %-20s\n", "Input", VALUE_WIDTH, "Value", "Result");
    print_separator(out, '-', TABLE_WIDTH);
    
    char val_buf[32], res_buf[32];
    for (int i = 0; i < N; i++) {
        InputType arg = generate_special_int_scalar<InputType>(i);
        ResultType result = FuncTag{}(arg);
        
        // Format integer input value
        if constexpr (std::is_same_v<InputType, int32_t>) {
            snprintf(val_buf, sizeof(val_buf), "%d", arg);
        } else if constexpr (std::is_same_v<InputType, int64_t>) {
            snprintf(val_buf, sizeof(val_buf), "%lld", (long long)arg);
        } else if constexpr (std::is_same_v<InputType, uint32_t>) {
            snprintf(val_buf, sizeof(val_buf), "%u", arg);
        } else if constexpr (std::is_same_v<InputType, uint64_t>) {
            snprintf(val_buf, sizeof(val_buf), "%llu", (unsigned long long)arg);
        }
        
        // Format FP result
        format_special_result(result, res_buf, sizeof(res_buf));
        
        // Get appropriate name based on type
        const char* name = nullptr;
        if constexpr (std::is_same_v<InputType, int32_t>) {
            name = get_special_int32_name(i);
        } else if constexpr (std::is_same_v<InputType, int64_t>) {
            name = get_special_int64_name(i);
        } else if constexpr (std::is_same_v<InputType, uint32_t>) {
            name = get_special_uint32_name(i);
        } else if constexpr (std::is_same_v<InputType, uint64_t>) {
            name = get_special_uint64_name(i);
        }
        
        fprintf(out, "%-5s | %*s | %-20s\n", name, VALUE_WIDTH, val_buf, res_buf);
    }
    
    // Print legend for integer types
    print_separator(out, '-', TABLE_WIDTH);
    if constexpr (std::is_same_v<InputType, int32_t>) {
        fprintf(out, "Legend: +MAX = 2^31-1 (int max), -MAX = -2^31 (int min), half = 2^30\n");
    } else if constexpr (std::is_same_v<InputType, int64_t>) {
        fprintf(out, "Legend: +MAX = 2^63-1 (llong max), -MAX = -2^63 (llong min), 1T = 10^12\n");
    } else if constexpr (std::is_same_v<InputType, uint32_t>) {
        fprintf(out, "Legend: MAX = 2^32-1 (uint max), half = 2^31, MSB = 2^31 (MSB set)\n");
    } else if constexpr (std::is_same_v<InputType, uint64_t>) {
        fprintf(out, "Legend: MAX = 2^64-1 (ullong max), half = 2^63, 1T = 10^12\n");
    }
    print_separator(out, '-', TABLE_WIDTH);
}

// ============================================================================
// Print Special Values Table - dispatches based on function arity and types
// ============================================================================
template<typename FuncTag, typename TestType>
inline void print_special_values_table(FILE* out, const char* func_name,
                                       const char* method_name = nullptr,
                                       const char* dataset_name = nullptr)
{
    using InputType = typename FuncTag::input_type;
    using ResultType = typename FuncTag::result_type;
    constexpr int arity = detect_arity<FuncTag>();
    
    // Use special integer input table for int2mp, ll2mp, uint2mp, ull2mp
    if constexpr (std::is_integral_v<InputType>) {
        print_special_values_table_int_input<FuncTag, TestType>(out, func_name, method_name, dataset_name);
        return;
    } 
    // Use special integer output table for mp2int, mp2ll, mp2uint, mp2ull (unary functions only)
    // Binary comparison functions (eq, ne, gt, etc.) return int but take 2 FP args - use 2D table
    else if constexpr (std::is_integral_v<ResultType> && arity == 1) {
        print_special_values_table_int_output<FuncTag, TestType>(out, func_name, method_name, dataset_name);
    }
    else if constexpr (arity == 1) {
        print_special_values_table_1d<FuncTag, TestType>(out, func_name, method_name, dataset_name);
    } 
    else if constexpr (arity == 2) {
        print_special_values_table_2d<FuncTag, TestType>(out, func_name, method_name, dataset_name);
    } 
    else {
        // For 3+ argument functions, just print a note
        fprintf(out, "\n");
        print_separator(out, '=');
        fprintf(out, "Special Values Table: %s (%d-argument) - not supported\n", func_name, arity);
        fprintf(out, "====================================================================================\n\n");
    }
}

#endif // __TS_PRINT_HPP__

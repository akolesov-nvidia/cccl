#ifndef TS_FIXED_HPP
#define TS_FIXED_HPP

namespace ts
{
    template<typename T, size_t nargs>
    struct fixed_t;

    template<typename T>
    struct fixed_t<T, 1>
    {
        T arg1, arg2 = 0, arg3 = 0;
        fixed_t(const T& arg1s) : arg1(ts::bit_cast<T>(arg1s)) {}
        T operator=(const T& other) { arg1 = other; return arg1; }    
    };

    template<typename T>
    struct fixed_t<T, 2>
    {
        T arg1, arg2, arg3 = 0;
        fixed_t(const T& arg1s, const T& arg2s) : arg1(ts::bit_cast<T>(arg1s)), 
                                                  arg2(ts::bit_cast<T>(arg2s)) {}
        T operator=(const T& other) { arg1 = other; return arg1; }
    };

    template<typename T>
    struct fixed_t<T, 3>
    {
        T arg1, arg2, arg3;
        fixed_t(const T& arg1s, const T& arg2s, const T& arg3s) : arg1(ts::bit_cast<T>(arg1s)), 
                                                                  arg2(ts::bit_cast<T>(arg2s)), 
                                                                  arg3(ts::bit_cast<T>(arg3s)) {}
        T operator=(const T& other) { arg1 = other; return arg1; }
    };

    template<typename T>
    struct fixed_t<T, 4>
    {
        T arg1, arg2, arg3, arg4;
        fixed_t(const T& arg1s, const T& arg2s, const T& arg3s, const T& arg4s) : arg1(ts::bit_cast<T>(arg1s)), 
                                                                                  arg2(ts::bit_cast<T>(arg2s)), 
                                                                                  arg3(ts::bit_cast<T>(arg3s)),
                                                                                  arg4(ts::bit_cast<T>(arg4s)) {}
        T operator=(const T& other) { arg1 = other; return arg1; }
    };

    static fixed_t<uint64_t, 2> add_fixed[] = {
        {0x7ff8000000000001, 0x7ff8000000000000},
        {0xfffcba93d9a2fa6c, 0x7ffded6e5f557125},
        {0xfff35ec64254a1a3, 0xfff68e2b0dd6a98e},
        {0x00142609774b3bda, 0x8020c56a5d77faeb},
        {0x8025ed558765ac25, 0x001d3ac9362ac7b7},
    };
    static fixed_t<uint64_t, 2> dadd_fixed[] = {
        {0x7ff8000000000001, 0x7ff8000000000000},
        {0xfffcba93d9a2fa6c, 0x7ffded6e5f557125},
        {0xfff35ec64254a1a3, 0xfff68e2b0dd6a98e},
        {0x00142609774b3bda, 0x8020c56a5d77faeb},
        {0x8025ed558765ac25, 0x001d3ac9362ac7b7},
    };
    static fixed_t<uint64_t, 2> mul_fixed[] = {
        {0xc081680000000000, 0x3fe62e42fefa39ef},
        {0xc083480000000000, 0x3fe62e42fefa39ef},
        {0x4083480000000000, 0x3fe62e42fefa39ef},
        {0x4081680000000000, 0x3fe62e42fefa39ef},
        {0x7ff8000000000001, 0x7ff8000000000000},
        {0x3fd8000000000002, 0xc008000000000004},
        {0x8873b8344e688c1d, 0x91d7df5c6f0c8ca9},
        {0xbfd63756d5289599, 0x0010000000000000},
    };
    static fixed_t<uint64_t, 2> dmul_fixed[] = {
        {0xc081680000000000, 0x3fe62e42fefa39ef},
        {0xc083480000000000, 0x3fe62e42fefa39ef},
        {0x4083480000000000, 0x3fe62e42fefa39ef},
        {0x4081680000000000, 0x3fe62e42fefa39ef},
        {0x7ff8000000000001, 0x7ff8000000000000},
        {0x3fd8000000000002, 0xc008000000000004},
        {0x8873b8344e688c1d, 0x91d7df5c6f0c8ca9},
        {0xbfd63756d5289599, 0x0010000000000000},
    };
    static fixed_t<uint64_t, 2> sub_fixed[] = {
        {0x7ff8000000000000, 0x7ff8000000000001},
        {0xfffcba93d9a2fa6c, 0x7ffded6e5f557125},
        {0xfff35ec64254a1a3, 0xfff68e2b0dd6a98e},
    };
    static fixed_t<uint64_t, 2> dsub_fixed[] = {
        {0x7ff8000000000000, 0x7ff8000000000001},
        {0xfffcba93d9a2fa6c, 0x7ffded6e5f557125},
        {0xfff35ec64254a1a3, 0xfff68e2b0dd6a98e},
    };
    static fixed_t<uint64_t, 2> div_fixed[] = {
        {0x7ff8000000000000, 0x7ff8000000000001},
        {0xfffcba93d9a2fa6c, 0x7ffded6e5f557125},
        {0xfff35ec64254a1a3, 0xfff68e2b0dd6a98e},
    };
    static fixed_t<uint64_t, 2> ddiv_fixed[] = {
        {0x7ff8000000000000, 0x7ff8000000000001},
        {0xfffcba93d9a2fa6c, 0x7ffded6e5f557125},
        {0xfff35ec64254a1a3, 0xfff68e2b0dd6a98e},
    };
    static fixed_t<uint64_t, 3> fma_fixed[] = {
        {0xda7a5d240467f51b, 0x7ff1864a926d1905, 0xfffcd6a2c124c2ba},
        {0xfff4b4006892d992, 0x7ff2778c82e7fea2, 0xb3ccc9333c9d79e7},
        {0xfffcba93d9a2fa6c, 0x7ffded6e5f557125, 0x268f8a49afcb7853},
        {0xed24d723b6b117d6, 0xfff35ec64254a1a3, 0xfff68e2b0dd6a98e},
        {0x7fff87916d2e62cf, 0x7fffa9088ecf22ac, 0xb4ac513e12bad036},
        {0x377c4eac0ec8d9e7, 0xfff37d39748bd851, 0x7ff931cb82ed0a94},
        {0xf61887aa989d1652, 0xf9375518d0ea9e5a, 0x3d4badd350e1c712},
        {0xc70b9d23f22e4f3c, 0xfecd8e85cbd29e54, 0xff2716483759f420},
    };
    static fixed_t<uint64_t, 3> dfma_fixed[] = {
        {0xda7a5d240467f51b, 0x7ff1864a926d1905, 0xfffcd6a2c124c2ba},
        {0xfff4b4006892d992, 0x7ff2778c82e7fea2, 0xb3ccc9333c9d79e7},
        {0xfffcba93d9a2fa6c, 0x7ffded6e5f557125, 0x268f8a49afcb7853},
        {0xed24d723b6b117d6, 0xfff35ec64254a1a3, 0xfff68e2b0dd6a98e},
        {0x7fff87916d2e62cf, 0x7fffa9088ecf22ac, 0xb4ac513e12bad036},
        {0x377c4eac0ec8d9e7, 0xfff37d39748bd851, 0x7ff931cb82ed0a94},
        {0xf61887aa989d1652, 0xf9375518d0ea9e5a, 0x3d4badd350e1c712},
        {0xc70b9d23f22e4f3c, 0xfecd8e85cbd29e54, 0xff2716483759f420},
    };
    static fixed_t<uint64_t, 3> mad_fixed[] = {
        {0xda7a5d240467f51b, 0x7ff1864a926d1905, 0xfffcd6a2c124c2ba},
        {0xfff4b4006892d992, 0x7ff2778c82e7fea2, 0xb3ccc9333c9d79e7},
        {0xfffcba93d9a2fa6c, 0x7ffded6e5f557125, 0x268f8a49afcb7853},
        {0xed24d723b6b117d6, 0xfff35ec64254a1a3, 0xfff68e2b0dd6a98e},
        {0x7fff87916d2e62cf, 0x7fffa9088ecf22ac, 0xb4ac513e12bad036},
    };
    static fixed_t<uint64_t, 4> dot_fixed[] = {
    };
    static fixed_t<uint64_t, 4> cmul_fixed[] = {
    };
    static fixed_t<uint64_t, 1> exp_fixed[] = {
        {0xc0782657c6550985},
        {0xc07ab65eb826ed7c},
        {0x407ab959e193479f},
        {0x40782576cfa37825},
        
    };
    static fixed_t<uint64_t, 1> poly4_fixed[] = {
        {0xfff0000000000000},
        {0xffefffffffffffff},
        {0x7fefffffffffffff},
        {0x7ff0000000000000},
        {0xce4c0b08101c88d3},
        {0x7dc00e8e59dae16a},
        {0xe05ebfc37af70747},
        {0x755d0ea91c20dbde}
    };
    static fixed_t<uint64_t, 1> polyf4_fixed[] = {
        {0xfff0000000000000},
        {0xffefffffffffffff},
        {0x7fefffffffffffff},
        {0x7ff0000000000000},
        {0xce4c0b08101c88d3},
        {0x7dc00e8e59dae16a},
        {0xe05ebfc37af70747},
        {0x755d0ea91c20dbde}
    };
    static fixed_t<uint64_t, 1> poly8_fixed[] = {
        {0xfff0000000000000},
        {0xffefffffffffffff},
        {0x7fefffffffffffff},
        {0x7ff0000000000000},
        {0xce4c0b08101c88d3},
        {0x7dc00e8e59dae16a},
        {0xe05ebfc37af70747},
        {0x755d0ea91c20dbde}
    };  
    static fixed_t<uint64_t, 1> polyf8_fixed[] = {
        {0xfff0000000000000},
        {0xffefffffffffffff},
        {0x7fefffffffffffff},
        {0x7ff0000000000000},
        {0xce4c0b08101c88d3},
        {0x7dc00e8e59dae16a},
        {0xe05ebfc37af70747},
        {0x755d0ea91c20dbde}
    }; 
    static fixed_t<uint64_t, 1> polyu8_fixed[] = {
        {0xfff0000000000000},
        {0xffefffffffffffff},
        {0x7fefffffffffffff},
        {0x7ff0000000000000},
    };
    static fixed_t<uint64_t, 1> sqrt_fixed[] = {
        {0x7ff8000000000000},
        {0x7ff8000000000000},
        {0x7ff8000000000000},
        
    };
    static fixed_t<uint64_t, 1> rsqrt_fixed[] = {
        {0x7ff8000000000000},
        {0x7ff8000000000000},
        {0x7ff8000000000000},
        
    };
    static fixed_t<uint64_t, 1> dsqrt_fixed[] = {
        {0x7ff8000000000000},
        {0x7ff8000000000000},
        {0x7ff8000000000000},
        
    };
} // end of namespace ts

#endif


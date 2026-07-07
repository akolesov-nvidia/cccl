/*
    copyable.cpp - Unit Test for Trivial Copyability and Volatile Value Preservation
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2026

    This test verifies that fpmp2_t types are trivially copyable and that
    volatile reads/writes preserve values correctly. A compile-time static_assert
    checks trivial copyability, and a runtime check confirms that a value survives
    a round-trip through a volatile object.
*/

#include <cuda/fpmp>

#include <cstdio>
#include <type_traits>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

int main()
{
    static_assert(std::is_trivially_copyable<fp32mp2>::value, "fp32mp2 must be trivially copyable");

    // Create a volatile object
    volatile fp32mp2 vx[1];
    
    // Create a non-volatile object and initialize it with a value
    fp32mp2 x[1] = { fp32mp2(1.0e+20) } ;

    // Assign the non-volatile object to the volatile object
    vx[0]          = x[0];

    // Print both objects
    printf("x[0]  = %f\n", (float)x[0]);
    printf("vx[0] = %f\n", (float)vx[0]);

    // Check if the volatile object has the same value as the non-volatile object
    if (vx[0] != x[0]) 
    {
        printf("ERROR: vx[0] != x[0]\n");
        return 1;
    }
    else 
    {
        printf("PASS: vx[0] == x[0]\n");
    }

    return 0;
}

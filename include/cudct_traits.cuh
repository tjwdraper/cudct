#ifndef _CUDCT_TRAITS_CUH_
#define _CUDCT_TRAITS_CUH_

#include <cufft.h>
#include <mex.h>

template <typename T>
struct cudct_traits;

template<>
struct cudct_traits<float> {
    using cudctComplex = cufftComplex;
    static constexpr cufftType fftType = CUFFT_C2C;
    static constexpr mxClassID mxType = mxSINGLE_CLASS;
};

template <>
struct cudct_traits<double> {
    using cudctComplex = cufftDoubleComplex;
    static constexpr cufftType fftType = CUFFT_Z2Z;
    static constexpr mxClassID mxType = mxDOUBLE_CLASS;
};

#endif
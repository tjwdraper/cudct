#ifndef _CUDA_DCT_CUH_
#define _CUDA_DCT_CUH_

#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

template <typename T>
struct cudct_traits;

template<>
struct cudct_traits<float> {
    using cudctComplex = cufftComplex;
    using cudctExec = cufftResult (*)(cufftHandle, cudctComplex*, cudctComplex*, int);
    static constexpr cudctExec exec = &cufftExecC2C;
    static constexpr cufftType fftType = CUFFT_C2C;
};

template <>
struct cudct_traits<double> {
    using cudctComplex = cufftDoubleComplex;
    using cudctExec = cufftResult (*)(cufftHandle, cudctComplex*, cudctComplex*, int);
    static constexpr cudctExec exec = &cufftExecZ2Z;
    static constexpr cufftType fftType = CUFFT_Z2Z;
};

template<typename T>
class cudct {
    public:
        // Constructors and deconstructors
        cudct(const std::vector<size_t>& dims);
        ~cudct();

        // DCT methods
        void dct(T* const output, const T* const input);
        void idct(T* const output, const T* const input);

    private:
        void set_freq_from_real(const T* const input, const int d);
        void set_freq_from_coefs(const T* const input, const int d);
        void multiply_weights(T* const output, int d) const;
        void rearrange_coefs(T* const output, const int d) const;
        void shift_dimensions(T* const output, const T* const input, std::size_t offset) const;

        std::vector<cufftHandle> _plans;
        std::vector<typename cudct_traits<T>::cudctComplex*> _weights;
        
        std::vector<std::size_t> _dims;
        std::size_t* _dims_d; // Device copy of _dims.data()

        std::size_t _ndim;
        std::size_t _size;

        typename cudct_traits<T>::cudctComplex* _freq;
        T* _tmp;
};

#endif
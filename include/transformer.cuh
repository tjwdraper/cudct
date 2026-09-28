#ifndef _TRANSFORMER_CUH_
#define _TRANSFORMER_CUH_

#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

template <typename T>
struct cudct_traits;

template<>
struct cudct_traits<float> {
    using cudctComplex = cufftComplex;
    static constexpr cufftType fftType = CUFFT_C2C;
};

template <>
struct cudct_traits<double> {
    using cudctComplex = cufftDoubleComplex;
    static constexpr cufftType fftType = CUFFT_Z2Z;
};

template<typename T>
class transformer {
    public:
        // Constructors and deconstructors
        transformer(const std::vector<size_t>& dims);
        ~transformer();

        // DCT methods
        void dct(T* const output, const T* const input);
        void idct(T* const output, const T* const input);

    private:
        using traits = cudct_traits<T>;

        void set_freq_from_real(const T* const, const int d);
        void multiply_weights(T* const output, int d) const;
        void shift_dimensions(T* const output, const T* const input, std::size_t offset) const;

        std::vector<cufftHandle> _plans;
        std::vector<traits::cudctComplex*> _weights;
        
        std::vector<std::size_t> _dims;
        std::size_t* _dims_d; // Device copy of _dims.data()

        std::size_t _ndim;
        std::size_t _size;

        traits::cudctComplex* _freq;
        T* _tmp;
};

#endif
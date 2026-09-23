#ifndef _TRANSFORMER_CUH_
#define _TRANSFORMER_CUH_

#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>


class transformer {
    public:
        // Constructors and deconstructors
        transformer(const std::vector<size_t>& dims);
        ~transformer();

        // DCT methods
        void dct(double* output, const double* const input);

    private:
        void set_freq_from_real(const double* const);
        void multiply_weights(double* output, int d) const;

        std::vector<cufftHandle> _plans;
        std::vector<cufftDoubleComplex*> _weights;
        
        std::vector<size_t> _dims;
        // std::vector<size_t> _strides;

        size_t _ndim;
        size_t _size;

        cufftDoubleComplex* _freq;
        double* _tmp;
};

#endif
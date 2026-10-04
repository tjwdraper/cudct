#include "include/cudct.cuh"

#include <stdexcept>
#include <climits>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <cmath>

///////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ 
void set_weights_kernel(typename cudct_traits<T>::cudctComplex* ww, const std::size_t siz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > siz - 1) {
        return;
    }

    ww[i].x = 2*cos(M_PI * i * -1.0 / (2.0 * siz)) / sqrt(2.0 * siz); // TODO: if T=float, change cos/sin/sqrt -> cosf/sinf/sqrtf.
    ww[i].y = 2*sin(M_PI * i * -1.0 / (2.0 * siz)) / sqrt(2.0 * siz);

    // Set the DC frequency separately.
    if (i == 0) {
        ww[i].x = ww[i].x / sqrt(2.0); // TODO: CUDA has automatic inverse square root function.
        ww[i].y = ww[i].y / sqrt(2.0);
    }
}

template <typename T>
__global__ 
void set_freq_from_real_kernel(typename cudct_traits<T>::cudctComplex* const freq, 
                               const T* const input, 
                               std::size_t cx, std::size_t Nx, std::size_t N) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    std::size_t i = idx % Nx;
    std::size_t j = idx / Nx;

    std::size_t i_shift = (i < cx ? 2*i : 2*Nx - 2*i - 1);

    freq[i + j * Nx].x = input[i_shift + j * Nx];
    freq[i + j * Nx].y = 0.0;
}

template<typename T>
__global__
void set_freq_from_coefs_kernel(typename cudct_traits<T>::cudctComplex* const freq,
                                const T* const input,
                                const typename cudct_traits<T>::cudctComplex* const weights,
                                std::size_t Nx, std::size_t N) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    int i = idx % Nx;

    freq[idx].x = input[idx] * weights[i].x;
    freq[idx].y = input[idx] * weights[i].y;
}

template <typename T>
__global__ 
void multiply_weights_kernel(T* const output, 
                             const typename cudct_traits<T>::cudctComplex* const freq, 
                             const typename cudct_traits<T>::cudctComplex* const weights, 
                             std::size_t Nx, std::size_t N) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    int i = idx % Nx;

    output[idx] = freq[idx].x * weights[i].x - freq[idx].y * weights[i].y;
}

template <typename T>
__global__
void rearrange_coefs_kernel(T* const output,
                            const typename cudct_traits<T>::cudctComplex* const freq,
                            std::size_t Nx,
                            std::size_t N) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    std::size_t i = idx % Nx;
    std::size_t j = idx / Nx;

    std::size_t i_shift = (i % 2 == 0 ? i / 2 : Nx - (i-1)/2 - 1);

    output[i + j * Nx] = freq[i_shift + j * Nx].x;
}

template <typename T>
__global__
void shift_dimensions_kernel(
    T* const output,
    const T* const input,
    std::size_t N,
    const std::size_t* const dims,
    std::size_t ndim,
    std::size_t offset
) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    std::size_t remainder = idx;
    std::size_t idx_shift = 0;

    for (std::size_t d = 0; d < ndim; ++d) {
        // Dimension d in the currently rotated array.
        std::size_t input_dim = (d + offset) % ndim;
        std::size_t dim = dims[input_dim];

        // Coordinate in this dimension.
        std::size_t coord = remainder % dim;
        remainder /= dim;

        // This dimension moves one position to the left.
        std::size_t output_dim = (d + ndim - 1) % ndim;

        // Calculate its stride in the output array.
        std::size_t output_stride = 1;

        for (std::size_t k = 0; k < output_dim; ++k) {
            std::size_t output_input_dim = (k + 1 + offset) % ndim;
            output_stride *= dims[output_input_dim];
        }

        idx_shift += coord * output_stride;
    }

    output[idx_shift] = input[idx];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Private methods
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void cudct<T>::set_freq_from_real(const T* const input, const int d) {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((_size + threadsPerBlock - 1) / threadsPerBlock);

    const std::size_t nx = _dims[d];
    const std::size_t cx = nx / 2 + (nx % 2 == 0 ? 0 : 1);

    set_freq_from_real_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(_freq, input, cx, nx, _size);
}

template <typename T>
void cudct<T>::set_freq_from_coefs(const T* const input, const int d) {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((_size + threadsPerBlock - 1) / threadsPerBlock);

    set_freq_from_coefs_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(_freq, input, _weights[d], _dims[d], _size);
}

template <typename T>
void cudct<T>::multiply_weights(T* const output, int d) const {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((_size + threadsPerBlock - 1) / threadsPerBlock);

    multiply_weights_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(output, _freq, _weights[d], _dims[d], _size);
}

template <typename T>
void cudct<T>::rearrange_coefs(T* const output, const int d) const {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((_size + threadsPerBlock - 1) / threadsPerBlock);

    const std::size_t nx = _dims[d];

    rearrange_coefs_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(output, _freq, nx, _size);
}

template <typename T>
void cudct<T>::shift_dimensions(T* const output, const T* const input, std::size_t offset) const {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((_size + threadsPerBlock - 1) / threadsPerBlock);

    shift_dimensions_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(output, input, _size, _dims_d, _ndim, offset);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public methods
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
cudct<T>::cudct(const std::vector<std::size_t>& dims) : _dims(dims) {
    // Check for zero dimensions
    for (std::size_t d : _dims) {
        if (d == 0)
            throw std::runtime_error("Dimensions must be non-zero.");
    }

    // Number of dimensions
    _ndim = dims.size();
    if (_ndim <= 0)
        throw std::runtime_error("cudct::cudct(const std::vector<std::size_t>) has to contain input with at least one entry.");

    // Set strides and size of the input matrix
    _size = dims[0];
    for (std::size_t d = 1; d < _ndim; ++d)
        _size *= dims[d];

    // Copy dimensions from host to device
    cudaMalloc((void**)&_dims_d, _ndim*sizeof(std::size_t));
    cudaMemcpy(_dims_d, _dims.data(), _ndim*sizeof(std::size_t), cudaMemcpyHostToDevice);

    // Create cufft plans to transform along each dimension
    _plans.resize(_ndim);
    for (std::size_t d = 0; d < _ndim; ++d) {
        int n = static_cast<int>(_dims[d]);
        int batch = static_cast<int>(_size / n);

        cufftResult results = cufftPlanMany(
            &_plans[d],
            1,
            &n,
            nullptr, 1, n,
            nullptr, 1, n,
            cudct_traits<T>::fftType,
            batch
        );

        if (results != CUFFT_SUCCESS)
            throw std::runtime_error("Could not create cuFFT plan");
    }

    // Allocate and initialize weight vectors
    _weights.resize(_ndim);
    for (std::size_t d = 0; d < _ndim; ++d) {
        cudaMalloc((void**)&_weights[d], _dims[d]*sizeof(typename cudct_traits<T>::cudctComplex));

        constexpr int threadsPerBlock = 256;
        int blocksPerGrid = static_cast<int>((_dims[d] + threadsPerBlock - 1) / threadsPerBlock );

        set_weights_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(_weights[d], _dims[d]);
    }

    // Initialize auxiliary arrays for intermediate results
    cudaMalloc((void**)&_tmp, _size*sizeof(T));
    cudaMalloc((void**)&_freq, _size*sizeof(typename cudct_traits<T>::cudctComplex));

    // Sync
    cudaDeviceSynchronize();
}

template <typename T>
cudct<T>::~cudct() {
    for (auto plan : _plans)
        cufftDestroy(plan);

    for (auto weights : _weights)
        cudaFree(weights);

    cudaFree(_dims_d);
    cudaFree(_tmp);
    cudaFree(_freq);
}

// DCT
template <typename T>
void cudct<T>::dct(T* const output, const T* const input) {
    // Iterative over dimensions
    for (std::size_t d = 0; d < _ndim; ++d) {
        if (d == 0)
            cudct<T>::set_freq_from_real(input, d);
        else
            cudct<T>::set_freq_from_real(output, d);

        cudct_traits<T>::exec(_plans[d], _freq, _freq, CUFFT_FORWARD);

        cudct<T>::multiply_weights(_tmp, d);

        cudct<T>::shift_dimensions(output, _tmp, d);
    }
}

template <typename T>
void cudct<T>::idct(T* const output, const T* const input) {
    // Iterate over dimensions
    for (std::size_t d = 0; d < _ndim; ++d) {
        if (d == 0)
            cudct<T>::set_freq_from_coefs(input, d);
        else
            cudct<T>::set_freq_from_coefs(output, d);

        cudct_traits<T>::exec(_plans[d], _freq, _freq, CUFFT_FORWARD);

        cudct<T>::rearrange_coefs(_tmp, d);

        cudct<T>::shift_dimensions(output, _tmp, d);
    }
}

template class cudct<float>;
template class cudct<double>;
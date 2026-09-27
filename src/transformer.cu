#include "include/transformer.cuh"

#include <stdexcept>
#include <climits>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <cmath>

///////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ 
void set_weights_kernel(cufftDoubleComplex* ww, const size_t siz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > siz - 1) {
        return;
    }

    ww[i].x = 2*cos(M_PI * i * -1.0 / (2.0 * siz)) / sqrt(2.0 * siz);
    ww[i].y = 2*sin(M_PI * i * -1.0 / (2.0 * siz)) / sqrt(2.0 * siz);

    // Set the DC frequency separately.
    if (i == 0) {
        ww[i].x = ww[i].x / sqrt(2.0);
        ww[i].y = ww[i].y / sqrt(2.0);
    }
}

__global__ 
void set_freq_kernel(cufftDoubleComplex* freq, const double* const input, std::size_t cx, std::size_t Nx, std::size_t N) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    std::size_t i = idx % Nx;
    std::size_t j = idx / Nx;

    std::size_t i_shift = (i < cx ? 2*i : 2*Nx - 2*i - 1);

    freq[i + j * Nx].x = input[i_shift + j * Nx];
    freq[i + j * Nx].y = 0.0;
}

__global__ 
void multiply_weights_kernel(double* const output, const cufftDoubleComplex* const freq, const cufftDoubleComplex* const weights, size_t Nx, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    int i = idx % Nx;

    output[idx] = freq[idx].x * weights[i].x - freq[idx].y * weights[i].y;
}

__global__ 
void shift_dimensions_kernel(double* const output, const double* const input, std::size_t N, const std::size_t* const dims, std::size_t ndim) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    std::size_t remainder = idx;
    std::size_t idx_shift = 0;
    std::size_t stride = 1;

    for (std::size_t d = 0; d < ndim; ++d) {
        // Coordinate in input dimension d
        std::size_t coord = remainder % dims[d];
        remainder /= dims[d];

        // Input dimension d becomes output dimension d-1.
        //
        // Input:
        //   [0, 1, 2, ..., ndim-1]
        //
        // Output:
        //   [1, 2, ..., ndim-1, 0]
        //
        std::size_t output_dim = (d + ndim - 1) % ndim;

        // Need the output stride for output_dim.
        std::size_t output_stride = 1;

        for (std::size_t k = 0; k < output_dim; ++k)
            output_stride *= dims[(k + 1) % ndim];

        idx_shift += coord * output_stride;
    }

    output[idx_shift] = input[idx];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Private methods
///////////////////////////////////////////////////////////////////////////////////////////////////
void transformer::set_freq_from_real(const double* const input, const int d) {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((_size + threadsPerBlock - 1) / threadsPerBlock);

    const std::size_t nx = _dims[d];
    const std::size_t cx = nx / 2 + (nx % 2 == 0 ? 0 : 1);

    set_freq_kernel<<<blocksPerGrid, threadsPerBlock>>>(_freq, input, cx, nx, _size);
}

void transformer::multiply_weights(double* const output, int d) const {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((_size + threadsPerBlock - 1) / threadsPerBlock);

    multiply_weights_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, _freq, _weights[d], _dims[d], _size);
}


void transformer::shift_dimensions(double* const output, const double* const input, const std::size_t* const dims) const {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((_size + threadsPerBlock - 1) / threadsPerBlock);

    shift_dimensions_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, input, _size, dims, _ndim);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public methods
///////////////////////////////////////////////////////////////////////////////////////////////////
transformer::transformer(const std::vector<size_t>& dims) : _dims(dims) {
    // Check for zero dimensions
    for (size_t d : _dims) {
        if (d == 0)
            throw std::runtime_error("Dimensions must be non-zero.");
    }

    // Number of dimensions
    _ndim = dims.size();
    if (_ndim <= 0)
        throw std::runtime_error("transformer::transformer(const std::vector<size_t>) has to contain input with at least one entry.");

    // Set strides and size of the input matrix
    _size = dims[0];
    for (std::size_t d = 1; d < _ndim; ++d)
        _size *= dims[d];

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
            CUFFT_Z2Z,
            batch
        );

        if (results != CUFFT_SUCCESS)
            throw std::runtime_error("Could not create cuFFT plan");
    }

    // Allocate and initialize weight vectors
    _weights.resize(_ndim);
    for (std::size_t d = 0; d < _ndim; ++d) {
        cudaMalloc((void**)&_weights[d], _dims[d]*sizeof(cufftDoubleComplex));

        constexpr int threadsPerBlock = 256;
        int blocksPerGrid = static_cast<int>((_dims[d] + threadsPerBlock - 1) / threadsPerBlock );

        set_weights_kernel<<<blocksPerGrid, threadsPerBlock>>>(_weights[d], _dims[d]);
    }

    // Initialize auxiliary arrays for intermediate results
    cudaMalloc((void**)&_tmp, _size*sizeof(double));
    cudaMalloc((void**)&_freq, _size*sizeof(cufftDoubleComplex));

    // Sync
    cudaDeviceSynchronize();
}

transformer::~transformer() {
    for (auto plan : _plans)
        cufftDestroy(plan);

    for (auto weights : _weights)
        cudaFree(weights);

    cudaFree(_tmp);
    cudaFree(_freq);
}

// DCT
void transformer::dct(double* output, const double* const input) {
    // Copy, then in-place transform
    cudaMemcpy(output, input, _size*sizeof(double), cudaMemcpyDeviceToDevice);

    // Create copy of the dimensions
    std::vector<size_t> dims = _dims;

    std::size_t* dims_d;
    cudaMalloc((void**)&dims_d, _ndim*sizeof(std::size_t));

    // Iterative over dimensions
    for (size_t d = 0; d < _ndim; ++d) {
        set_freq_from_real(output, d);

        cufftExecZ2Z(_plans[d], _freq, _freq, CUFFT_FORWARD);

        multiply_weights(output, d);

        // Shift dimensions
        cudaMemcpy(dims_d, dims.data(), _ndim*sizeof(std::size_t), cudaMemcpyHostToDevice);

        shift_dimensions(_tmp, output, dims_d);
        cudaMemcpy(output, _tmp, _size*sizeof(double), cudaMemcpyDeviceToDevice);

        std::rotate(dims.begin(), dims.begin() + 1, dims.end());
    }

    cudaFree(dims_d);
}

void transformer::idct(double* output, const double* const input) {

}
#include "include/transformer.cuh"

#include <stdexcept>
#include <climits>


// CUDA kernels
__global__ void set_weights_kernel(cufftDoubleComplex* ww, const size_t siz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > siz - 1) {
        return;
    }

    constexpr double PI = 3.14159265358979323846;

    ww[i].x = 2*cos(PI * i * -1.0 / (2.0 * siz)) / sqrt(2.0 * siz);
    ww[i].y = 2*sin(PI * i * -1.0 / (2.0 * siz)) / sqrt(2.0 * siz);

    // Set the DC frequency separately.
    if (i == 0) {
        ww[i].x = ww[i].x / sqrt(2.0);
        ww[i].y = ww[i].y / sqrt(2.0);
    }
}

__global__ void set_freq_kernel(
    cufftDoubleComplex* freq,
    const double* input,
    size_t size
) {
    size_t idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    freq[idx].x = input[idx];
    freq[idx].y = 0.0;
}

__global__ void multiply_weights_kernel(
    double* output,
    const cufftDoubleComplex* freq,
    const cufftDoubleComplex* weights,
    size_t n,
    size_t size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    size_t i = idx % n;

    output[idx] =
        freq[idx].x * weights[i].x
        - freq[idx].y * weights[i].y;
}

__global__ void shift_dimensions_kernel(
    double* output,
    const double* input,
    const size_t* old_dims,
    const size_t* new_dims,
    const size_t* old_strides,
    const size_t* new_strides,
    size_t ndim,
    size_t size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    // Decode index in new array into coordinates.
    size_t remaining = idx;

    // Coordinates in the new array.
    // new coordinate:
    // [y, z, ..., x]
    size_t old_idx = 0;

    for (size_t d = 0; d < ndim; ++d) {
        size_t coord = remaining / new_strides[d];
        remaining %= new_strides[d];

        // New axis d corresponds to old axis (d + 1) % ndim.
        size_t old_axis = (d + 1) % ndim;

        old_idx += coord * old_strides[old_axis];
    }

    output[idx] = input[old_idx];
}

// Helpers
void transformer::set_freq_from_real(const double* const input) {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((_size + threadsPerBlock - 1) / threadsPerBlock);

    set_freq_kernel<<<blocksPerGrid, threadsPerBlock>>>(_freq, input, _size);
}

void transformer::multiply_weights(double* output, int d) const {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((_size + threadsPerBlock - 1) / threadsPerBlock);

    multiply_weights_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, _freq, _weights[d], _dims[d], _size);
}

            shift_dimensions(_tmp, output, dims);


// Constructors and deconstructors
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
    // _strides.resize(_ndim);
    // _strides[0] = 1;
    for (std::size_t d = 1; d < _ndim; ++d) {
        // _strides[d] = _strides[d-1] * _dims[d-1]; // Column-major ordering
        _size *= dims[d];
    }

    // Create cufft plans to transform along each dimension
    _plans.resize(_ndim);
    for (std::size_t d = 0; d < _ndim; ++d) {
        int n = static_cast<int>(_dims[d]);
        
        int batch = static_cast<int>(_size / _dims[d]);

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
transformer::dct(double* output, const double* const input) {
    // Copy, then in-place transform
    cudaMemcpy(output, input, _size*sizeof(double), cudaMemcpyDeviceToDevice);

    // Create copy of the dimensions
    std::vector<size_t> dims = _dims;

    // Iterative over dimensions
    for (size_t d = 0; d < _ndim; ++d) {
        set_freq_from_real(output);

        cufftExecZ2Z(_plans[d], _freq, _freq, CUFFT_FORWARD);

        multiply_weights(output, d);

        if (d+1 < _ndim) {
            shift_dimensions(_tmp, output, dims);

            cudaMemcpy(output, _tmp, _size*sizeof(double), cudaMemcpyDeviceToDevice);

            std::rotate(dims.begin(), dims.begin() + 1, dims.end());
        }
    }
}
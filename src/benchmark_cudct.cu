#include <cstring>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <map>

#include "include/cudct.cuh"

#include <vector>
#include <random>

std::vector<std::size_t> squeeze_dimensions(const std::size_t* const dims, const std::size_t ndim) {
    std::vector<std::size_t> dims_t;
    for (std::size_t d = 0; d < ndim; ++d) {
        if (dims[d] == 1)
            continue;
        dims_t.push_back(static_cast<std::size_t>(dims[d]));
    }
    return dims_t;
}

int main() {
    // Create dimensions
    const int ndim = 3;

    const std::size_t dims[ndim] = {256, 256, 256};
    const std::size_t numel = std::accumulate(dims, dims+ndim, 1, std::multiplies<std::size_t>{});

    // Create an instance of an engine
    std::random_device rnd_device;
    // Specify the engine and distribution
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_real_distribution<double> dist {0.0, 100.0};

    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    std::vector<double> input_h(numel);
    std::generate(input_h.begin(), input_h.end(), gen);

    // Create output vector
    std::vector<double> output_h(numel);

    // Allocate memory on the device
    double *input_d, *output_d;
    cudaMalloc((void**)&input_d, numel*sizeof(double));
    cudaMalloc((void**)&output_d, numel*sizeof(double));
    cudaMemcpy(input_d, input_h.data(), numel*sizeof(double), cudaMemcpyHostToDevice);

    // Benchmark with CUDA-DCT
    cudct<double> cudct(squeeze_dimensions(dims, ndim));

    cudct.dct(output_d, input_d);

    // Copy the result back to the host
    cudaMemcpy(output_h.data(), output_d, numel*sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}
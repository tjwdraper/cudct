#include <cstring>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <map>

#include <fftw3.h>

#include <vector>
#include <random>

int main() {
    // Create dimensions
    const int ndim = 3;
    const int dims[ndim] = {64, 64, 64};
    const int numel = std::accumulate(dims, dims+ndim, 1, std::multiplies<int>{});

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

    // Benchmark with FFTW
    fftw_r2r_kind* kinds = new fftw_r2r_kind[ndim];
    for (std::size_t d = 0; d < ndim; ++d)
        kinds[d] = FFTW_REDFT10;

    fftw_plan plan = fftw_plan_r2r(ndim, reinterpret_cast<const int*>(dims), input_h.data(), output_h.data(), kinds, FFTW_ESTIMATE);
    if (plan == nullptr)
        throw std::runtime_error("Could not create FFTW plan"); 

    fftw_execute(plan);

    fftw_destroy_plan(plan);
    delete[] kinds;

    return 0;
}


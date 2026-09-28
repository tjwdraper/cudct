#include <mex.h>
#include <cstring>
#include <numeric> 
#include <functional>
#include <stdexcept>

#include "include/transformer.cuh"
// #include "include/cudct_traits.cuh"

std::vector<std::size_t> squeeze_dimensions(const mwSize* const dims, const mwSize ndim) {
    std::vector<std::size_t> dims_t;
    for (mwSize d = 0; d < ndim; ++d) {
        if (dims[d] == 1)
            continue;
        dims_t.push_back(static_cast<std::size_t>(dims[d]));
    }
    return dims_t;
}

template<typename T>
void run_dct(T* const output, const T* const input, const char* const operation, const std::vector<std::size_t>& dims, const std::size_t numel) {
    // Allocate memory for input and output on device
    T* input_d;
    T* output_d;
    cudaMalloc((void**)&input_d, numel*sizeof(T));
    cudaMalloc((void**)&output_d, numel*sizeof(T));
    cudaMemcpy(input_d, input, numel*sizeof(T), cudaMemcpyHostToDevice);

    // Calculate the transform
    transformer<T> cudct(dims);
    if (strcmp(operation, "forward") == 0)
        cudct.dct(output_d, input_d);
    else if (strcmp(operation, "inverse") == 0)
        cudct.idct(output_d, input_d);

    // Move result from device to host
    cudaMemcpy(output, output_d, numel*sizeof(T), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(input_d);
    cudaFree(output_d);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check number of input and output variables
    if (nlhs != 1 || nrhs != 2)
        mexErrMsgTxt("Invalid number of input and output variables given");

    // Check data variable format
    if (mxIsComplex(prhs[0]))
        mexErrMsgTxt("First input must be a real matrix.");

    // Parse direction
    if (!mxIsChar(prhs[1]))
        mexErrMsgTxt("Second argument must be a character array.");
    
    char* operation = mxArrayToString(prhs[1]);
    if (operation == nullptr)
        mexErrMsgTxt("Could not convert to string.");

    // Load dimensions
    const mxArray* data = prhs[0];
    mwSize ndim = mxGetNumberOfDimensions(data);
    const mwSize* dims = mxGetDimensions(data);

    // Convert mwSize* to std::vector<std::size_t> and squeeze dimensions of length one:
    std::vector<std::size_t> dims_t = squeeze_dimensions(dims, ndim);
    std::size_t ndim_t = dims_t.size();
    const auto numel_t = std::accumulate(dims_t.cbegin(), dims_t.cend(), 1, std::multiplies<std::size_t>{});

    if (numel_t == 0)
        throw std::runtime_error("Must pass non-empty array to cudct.");

    // Distinguish between single and double precision
    if (mxIsDouble(prhs[0])) {
        plhs[0] = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
        double* output_h = static_cast<double*>(mxGetData(plhs[0]));
        double* input_h = static_cast<double*>(mxGetData(prhs[0]));

        run_dct<double>(output_h, input_h, operation, dims_t, numel_t);
    }
    else if (mxIsSingle(prhs[0])) {
        plhs[0] = mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS, mxREAL);
        float* output_h = static_cast<float*>(mxGetData(plhs[0]));
        float* input_h = static_cast<float*>(mxGetData(prhs[0]));

        run_dct<float>(output_h, input_h, operation, dims_t, numel_t);
    }

    // Free memory
    mxFree(operation);
}
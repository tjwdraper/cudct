#include <mex.h>
#include <cstring>
#include <numeric> 
#include <functional>
#include <stdexcept>

#include "include/transformer.cuh"

std::vector<std::size_t> squeeze_dimensions(const mwSize* const dims, const mwSize ndim) {
    std::vector<std::size_t> dims_t;
    for (mwSize d = 0; d < ndim; ++d) {
        if (dims[d] == 1)
            continue;
        dims_t.push_back(static_cast<std::size_t>(dims[d]));
    }
    return dims_t;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check number of input and output variables
    if (nlhs != 1 || nrhs != 2)
        mexErrMsgTxt("Invalid number of input and output variables given");

    // Check data variable format
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("First input must be a real double matrix.");

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

    // Get the input
    double* input_h = mxGetPr(data);
    double* input_d;
    cudaMalloc((void**)&input_d, numel_t*sizeof(double));
    cudaMemcpy(input_d, input_h, numel_t*sizeof(double), cudaMemcpyHostToDevice);

    // Create cudct object
    transformer cudct(dims_t);

    // Calculate the transform on device array
    double* output_d;
    cudaMalloc((void**)&output_d, numel_t*sizeof(double));

    if (strcmp(operation, "forward") == 0)
        cudct.dct(output_d, input_d);
    else if (strcmp(operation, "inverse") == 0)
        cudct.idct(output_d, input_d);
    else {
        mxFree(operation);
        mexErrMsgTxt("Operation must be 'forward' or 'inverse'.");
    }

    // Allocate memory for the output-II and execute plan
    plhs[0] = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
    double* output_h = mxGetPr(plhs[0]);

    cudaMemcpy(output_h, output_d, numel_t*sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(input_d);
    cudaFree(output_d);
    mxFree(operation);
}
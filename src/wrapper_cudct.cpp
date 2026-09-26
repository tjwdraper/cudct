#include <mex.h>
#include <cstring>

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
    if (ndim != 3) 
        mexErrMsgTxt("First argument must be a 3d-matrix");
    const mwSize* dims = mxGetDimensions(data);

    // Convert mwSize* to std::vector<std::size_t> and squeeze dimensions of length one:
    std::vector<std::size_t> dims_t = squeeze_dimensions(dims, ndim);
    std::size_t ndim_t = dims_t.size();

    // Get the input
    double* input = mxGetPr(data);

    // Allocate memory for the output-II and execute plan
    plhs[0] = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
    double* output = mxGetPr(plhs[0]);

    // Create cudct object
    transformer cudct(dims_t);

    if (strcmp(operation, "forward") == 0)
        cudct.dct(output, input);
    else if (strcmp(operation, "inverse") == 0)
        cudct.idct(output, input);
    else {
        mxFree(operation);
        mexErrMsgTxt("Operation must be 'forward' or 'inverse'.");
    }

    // Free memory
    mxFree(operation);
}
#include <mex.h>
#include <matrix.h>
#include <string>

#include "include/transformer.cuh"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check number of input and output variables
    if (nlhs != 1 || nrhs != 2)
        mexErrMsgTxt("Invalud number of input and output variables given");

    // Check data variable format
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("First input must be a real double matrix.");

    // Load dimensions
    const mxArray* data = prhs[0];
    mwSize ndim = mxGetNumberOfDimensions(data);
    if (ndim != 3) 
        mexErrMsgTxt("First argument must be a 3d-matrix");
    const mwSize* dims = mxGetDimensions(data);

    // Get the input
    double* input = mxGetPr(data);

    // Allocate memory for the output-II and execute plan
    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    double* output = mxGetPr(plhs[0]);

    // Parse direction
    if (!mxIsChar(prhs[1]))
        mexErrMsgTxt("Second argument must be a character array.");
    
    char* operation = mxArrayToString(prhs[1]);
    if (operation == nullptr)
        mexErrMsgTxt("Could not convert to string.");

    // Create plan
    if (strcmp(operation, "forward") == 0) {
        // ...
        
    }
    else if (strcmp(operation, "inverse") == 0) {
        mxFree(operation);
        mexErrMsgTxt("Inverse is not yet implemented.");
    }
    else {
        mxFree(operation);
        mexErrMsgTxt("Operation must be 'forward' or 'inverse'.");
    }

    // Free memory
    mxFree(operation);
}
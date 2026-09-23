#include <mex.h>
#include <matrix.h>
#include <string>
#include <fftw3.h>

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
    
    // Matlab stores 1d arrays as a 2d-array of size (N,1)
    if (ndim != 2) 
        mexErrMsgTxt("First argument must be a 1d array");
    const mwSize* dims = mxGetDimensions(data);
    if (dims[1] != 1)
        mexErrMsgTxt("First argument must be a 1d array");

    // Get the input
    double* input = mxGetPr(data);

    // Allocate memory for the output-II and execute plan
    plhs[0] = mxCreateNumericArray(1, dims, mxDOUBLE_CLASS, mxREAL);
    double* output = mxGetPr(plhs[0]);

    // Parse direction
    if (!mxIsChar(prhs[1]))
        mexErrMsgTxt("Second argument must be a character array.");
    
    char* operation = mxArrayToString(prhs[1]);
    if (operation == nullptr)
        mexErrMsgTxt("Could not convert to string.");

    // Create fftw plan:
    fftw_plan plan;
    if (strcmp(operation, "forward") == 0)
        plan = fftw_plan_r2r_1d(dims[0], input, output, FFTW_REDFT10, FFTW_ESTIMATE);
    else if (strcmp(operation, "inverse") == 0)
        plan = fftw_plan_r2r_1d(dims[0], input, output, FFTW_REDFT01, FFTW_ESTIMATE);
    else {
        mxFree(operation);
        mexErrMsgTxt("Operation must be 'forward' or 'inverse'.");
    }

    if (plan == nullptr)
        mexErrMsgTxt("Could not create FFTW plan.");

    // (I)DCT calculation
    fftw_execute(plan);

    // Free memory
    fftw_destroy_plan(plan);
    mxFree(operation);
}
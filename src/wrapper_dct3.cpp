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

    // Create fftw plan - Matlab uses column-major ordering, which means we need to swap dimensions if we want to match and 
    // compare with DCT implementation in Matlab.
    fftw_plan plan;
    if (strcmp(operation, "forward") == 0)
        plan = fftw_plan_r2r_3d(dims[2], dims[1], dims[0], input, output, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
    else if (strcmp(operation, "inverse") == 0)
        plan = fftw_plan_r2r_3d(dims[2], dims[1], dims[0], input, output, FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE);
    else {
        mxFree(operation);
        mexErrMsgTxt("Operation must be 'forward' or 'inverse'.");
    }

    if (plan == nullptr)
        mexErrMsgTxt("Could not create FFTW plan.");

    // (I)output calculation
    fftw_execute(plan);

    // Free memory
    fftw_destroy_plan(plan);
    mxFree(operation);
}
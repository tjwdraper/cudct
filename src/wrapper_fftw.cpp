#include <mex.h>
#include <cstring>
#include <fftw3.h>
#include <map>
#include <numeric>
#include <cmath>
#include <vector>

template <typename T>
void orthogonalize_forward(T* const output, const mwSize* dims, std::size_t ndim, std::size_t numel) {
    T sqrt2 = std::sqrt(static_cast<T>(2.0));
    T base = static_cast<T>(1.0) / (std::pow(static_cast<T>(2.0), static_cast<T>(ndim)) * std::sqrt(static_cast<T>(numel)));

    std::vector<T> factors(ndim+1);
    factors[0] = base;
    for (std::size_t i = 1; i <= ndim; ++i)
        factors[i] = factors[i-1] * sqrt2;

    for (std::size_t idx = 0; idx < numel; ++idx) {
        std::size_t remainder = idx;
        std::size_t nonzero = 0;

        for (std::size_t d = 0; d < ndim; ++d) {

            std::size_t coord = remainder % dims[d];
            remainder /= dims[d];

            if (coord != 0)
                ++nonzero;
        }

        output[idx] *= factors[nonzero];
    }
}

template <typename T>
void orthogonalize_inverse(T* const output, const mwSize* dims, std::size_t ndim, std::size_t numel) {
    T sqrt2 = std::sqrt(static_cast<T>(2.0));
    T base = static_cast<T>(1.0) / std::sqrt(static_cast<T>(numel));

    std::vector<T> factors(ndim+1);
    factors[0] = base;
    for (std::size_t i = 1; i <= ndim; ++i)
        factors[i] = factors[i-1] / sqrt2;

    for (std::size_t idx = 0; idx < numel; ++idx) {
        std::size_t remainder = idx;
        std::size_t nonzero = 0;

        for (std::size_t d = 0; d < ndim; ++d) {

            std::size_t coord = remainder % dims[d];
            remainder /= dims[d];

            if (coord != 0)
                ++nonzero;
        }

        output[idx] *= factors[nonzero];
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
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
    const mwSize numel = mxGetNumberOfElements(data);


    // Reverse dimensions for column major ordering
    int* dims_cm = new int[ndim];
    for (std::size_t d = 0; d < ndim; ++d)
        dims_cm[d] = static_cast<int>(dims[ndim - d - 1]);



    if (mxIsSingle(data)) {
        // Get the input
        float* input = (float*) mxGetData(data);

        // Allocate memory for the output
        plhs[0] = mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS, mxREAL);
        float* output = (float*) mxGetData(plhs[0]);

        // Create FFTW kinds
        fftwf_r2r_kind* kinds = new fftwf_r2r_kind[ndim];
        for (std::size_t d = 0; d < ndim; ++d)
            kinds[d] = (strcmp(operation, "forward") == 0 ? FFTW_REDFT10 : FFTW_REDFT01);

        // Initialize FFTW plan
        fftwf_plan plan;

        // Check for orthogonalization
        #ifdef MATLAB_ORTHOGONALIZE
            memcpy(output, input, numel*sizeof(float));
            if (strcmp(operation, "inverse") == 0)
                orthogonalize_inverse(output, dims, ndim, numel);

            plan = fftwf_plan_r2r(ndim, dims_cm, output, output, kinds, FFTW_ESTIMATE);
        #else
            plan = fftwf_plan_r2r(ndim, dims_cm, input, output, kinds, FFTW_ESTIMATE);
        #endif

        // Execute FFTW plan
        if (plan == nullptr)
            mexErrMsgTxt("Could not create FFTW plan");
        fftwf_execute(plan);

        // Free memory
        fftwf_destroy_plan(plan);
        delete[] kinds;

        // Orthogonalize output if forward transform
        #ifdef MATLAB_ORTHOGONALIZE
            if (strcmp(operation, "forward") == 0)
                orthogonalize_forward(output, dims, ndim, numel);
        #endif
    }
    else if (mxIsDouble(data)) {
        // Get the input
        double* input = (double*) mxGetData(data);

        // Allocate memory for the output
        plhs[0] = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
        double* output = (double*) mxGetData(plhs[0]);

        // Create FFTW kinds
        fftw_r2r_kind* kinds = new fftw_r2r_kind[ndim];
        for (std::size_t d = 0; d < ndim; ++d)
            kinds[d] = (strcmp(operation, "forward") == 0 ? FFTW_REDFT10 : FFTW_REDFT01);

        // Initialize FFTW plan
        fftw_plan plan;

        // Check for orthogonalization
        #ifdef MATLAB_ORTHOGONALIZE
            memcpy(output, input, numel*sizeof(double));
            if (strcmp(operation, "inverse") == 0)
                orthogonalize_inverse(output, dims, ndim, numel);

            plan = fftw_plan_r2r(ndim, dims_cm, output, output, kinds, FFTW_ESTIMATE);
        #else
            plan = fftw_plan_r2r(ndim, dims_cm, input, output, kinds, FFTW_ESTIMATE);
        #endif

        // Execute FFTW plan
        if (plan == nullptr)
            mexErrMsgTxt("Could not create FFTW plan");
        fftw_execute(plan);

        // Free memory
        fftw_destroy_plan(plan);
        delete[] kinds;

        // Orthogonalize output if forward transform
        #ifdef MATLAB_ORTHOGONALIZE
            if (strcmp(operation, "forward") == 0)
                orthogonalize_forward(output, dims, ndim, numel);
        #endif

    }
    
    // Free memory
    mxFree(operation);
    delete[] dims_cm;
        
}
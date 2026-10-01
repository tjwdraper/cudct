#include <mex.h>
#include <cstring>
#include <fftw3.h>
#include <map>
#include <numeric>
#include <cmath>

template <typename T>
void orthogonalize_forward(T* const output, const mwSize* dims, std::size_t ndim, std::size_t numel) {
    T base = 1.0 / (std::pow(2.0, static_cast<T>(ndim)) * std::sqrt(static_cast<T>(numel)));

    for (std::size_t idx = 0; idx < numel; ++idx) {
        std::size_t remainder = idx;
        std::size_t nonzero = 0;

        for (std::size_t d = 0; d < ndim; ++d) {

            std::size_t coord = remainder % dims[d];
            remainder /= dims[d];

            if (coord != 0)
                ++nonzero;
        }

        output[idx] *= base * std::pow(std::sqrt(2.0), static_cast<T>(nonzero));
    }
}

template <typename T>
void orthogonalize_inverse(T* const output, const mwSize* dims, std::size_t ndim, std::size_t numel) {
    T base = 1.0 / std::sqrt(static_cast<T>(numel));

    for (std::size_t idx = 0; idx < numel; ++idx) {
        std::size_t remainder = idx;
        std::size_t nonzero = 0;

        for (std::size_t d = 0; d < ndim; ++d) {

            std::size_t coord = remainder % dims[d];
            remainder /= dims[d];

            if (coord != 0)
                ++nonzero;
        }

        output[idx] *= base / std::pow(std::sqrt(2.0), static_cast<T>(nonzero));
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

        // Preprocess
        float* input_orthogonal = new float[numel];
        memcpy(input_orthogonal, input, numel*sizeof(float));
        // if (strcmp(operation, "inverse") == 0)
        //     orthogonalize_inverse(input_orthogonal, dims, ndim, numel);

        // Allocate memory for the output
        plhs[0] = mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS, mxREAL);
        float* output_orthogonal = (float*) mxGetData(plhs[0]);

        // Create FFTW kinds
        fftwf_r2r_kind* kinds = new fftwf_r2r_kind[ndim];
        for (std::size_t d = 0; d < ndim; ++d)
            kinds[d] = (strcmp(operation, "forward") == 0 ? FFTW_REDFT10 : FFTW_REDFT01);

        // Create FFTW plan
        fftwf_plan plan;
        plan = fftwf_plan_r2r(ndim, dims_cm, input_orthogonal, output_orthogonal, kinds, FFTW_ESTIMATE);

        if (plan == nullptr)
            mexErrMsgTxt("Could not create FFTW plan");

        // Execute plan
        fftwf_execute(plan);

        // Post-process
        // if (strcmp(operation, "forward") == 0)
        //     orthogonalize_forward(output_orthogonal, dims, ndim, numel);

        // Free memory
        fftwf_destroy_plan(plan);
        delete[] kinds;



    }
    else if (mxIsDouble(data)) {
        // Get the input
        double* input = (double*) mxGetData(data);

        // Preprocess
        double* input_orthogonal = new double[numel];
        memcpy(input_orthogonal, input, numel*sizeof(double));
        if (strcmp(operation, "inverse") == 0)
            orthogonalize_inverse(input_orthogonal, dims, ndim, numel);

        // Allocate memory for the output
        plhs[0] = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
        double* output_orthogonal = (double*) mxGetData(plhs[0]);

        // Create FFTW kinds
        fftw_r2r_kind* kinds = new fftw_r2r_kind[ndim];
        for (std::size_t d = 0; d < ndim; ++d)
            kinds[d] = (strcmp(operation, "forward") == 0 ? FFTW_REDFT10 : FFTW_REDFT01);

        // Create FFTW plan
        fftw_plan plan;
        plan = fftw_plan_r2r(ndim, dims_cm, input_orthogonal, output_orthogonal, kinds, FFTW_ESTIMATE);

        if (plan == nullptr)
            mexErrMsgTxt("Could not create FFTW plan");

        // Execute plan
        fftw_execute(plan);

        // Post-process
        if (strcmp(operation, "forward") == 0)
            orthogonalize_forward(output_orthogonal, dims, ndim, numel);

        // Free memory
        fftw_destroy_plan(plan);
        delete[] kinds;

    }
    
    // Free memory
    mxFree(operation);
    delete[] dims_cm;
        
}
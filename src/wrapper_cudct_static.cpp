#include <mex.h>
#include <cstring>
#include <numeric> 
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <map>

#include "include/transformer.cuh"

// Read strings from MATLAB and convert 
enum class Direction {CUDCT_FORWARD, CUDCT_INVERSE};

const std::map<std::string, Direction> direction_mapper {
    {"forward", Direction::CUDCT_FORWARD},
    {"inverse", Direction::CUDCT_INVERSE}
};

const std::map<std::string, mxClassID> precision_mapper {
    {"single", mxSINGLE_CLASS},
    {"double", mxDOUBLE_CLASS}
};

template <typename T>
T mxParseString(const mxArray* input, const std::map<std::string, T>& mapper) {
    if (!mxIsChar(input))
        mexErrMsgTxt("Input argument has to be a character array.");

    const char* input_c = mxArrayToString(input);
    std::string key(input_c);

    auto it = mapper.find(key);
    if (it == mapper.end())
        mexErrMsgIdAndTxt("Input argument %s is not a valid option.", input_c);

    return it->second;
}

// Create static object containing CUDCT configuration
struct cudct_config {
    void* cudct = nullptr;
    mwSize* dims = nullptr;
    mwSize ndim;
    mwSize numel;
    mxClassID precision;
};

static cudct_config config;
static bool configured = false;

// Helper functions
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
    if (nlhs == 0 && nrhs == 2 && !configured) {
        if (mxIsComplex(prhs[0]))
            mexErrMsgTxt("First input must be a real matrix.");

        if (!(mxIsDouble(prhs[0]) || mxIsSingle(prhs[0])))
            mexErrMsgTxt("First input must be of type double");

        if (mxGetNumberOfDimensions(prhs[0]) != 2)
            mexErrMsgTxt("First input must be a vector of size Nx1.");

        const mwSize* tmp = mxGetDimensions(prhs[0]);
        if (tmp[0] == 0)
            mexErrMsgTxt("First input cannot be empty.");

        if (tmp[1] != 1)
            mexErrMsgTxt("First input must be a vector of size Nx1.");

        config.ndim = mxGetNumberOfElements(prhs[0]);
        config.dims = new mwSize[config.ndim];

        double* mxDims = (double*) mxGetData(prhs[0]);
        for (std::size_t d = 0; d < config.ndim; ++d)
            config.dims[d] = static_cast<mwSize>(mxDims[d]);
    
        config.precision = mxParseString(prhs[1], precision_mapper);

        std::vector<std::size_t> dims_t = squeeze_dimensions(config.dims, config.ndim);
        config.numel = std::accumulate(dims_t.cbegin(), dims_t.cend(), 1, std::multiplies<std::size_t>{});
        
        if (config.precision == mxSINGLE_CLASS)
            config.cudct = new transformer<float>(dims_t);
        else if (config.precision == mxDOUBLE_CLASS)
            config.cudct = new transformer<double>(dims_t);
        else
            mexErrMsgTxt("Could not convert precision option to single or double.");

        configured = true;
    }
    else if (nlhs == 0 && nrhs == 2 && configured) {
        if (mxIsComplex(prhs[0]))
            mexErrMsgTxt("First input must be a real matrix.");
        
        if (mxGetClassID(prhs[0]) != config.precision)
            mexErrMsgTxt("Input precision does not match configured precision.");

        if (mxGetNumberOfDimensions(prhs[0]) != config.ndim)
            mexErrMsgTxt("First argument does not have same number of dimensions as when configured.");

        if (!std::equal(config.dims, config.dims+config.ndim, mxGetDimensions(prhs[0])))
            mexErrMsgTxt("First argument does not have the same dimensions as when configured");

        Direction dir = mxParseString(prhs[1], direction_mapper);

        if (config.precision == mxSINGLE_CLASS) {
            plhs[0] = mxCreateNumericArray(config.ndim, config.dims, config.precision, mxREAL);
            float* output_h = static_cast<float*>(mxGetData(plhs[0]));
            float* input_h = static_cast<float*>(mxGetData(prhs[0]));
            float* output_d;
            float* input_d;

            cudaMalloc((void**)&output_d, config.numel*sizeof(float));
            cudaMalloc((void**)&input_d, config.numel*sizeof(float));
            cudaMemcpy(input_d, input_h, config.numel*sizeof(float), cudaMemcpyHostToDevice);

            auto* cudct = static_cast<transformer<float>*>(config.cudct);
            if (dir == Direction::CUDCT_FORWARD)
                cudct->dct(output_d, input_d);
            else if (dir == Direction::CUDCT_INVERSE)
                cudct->idct(output_d, input_d);

            cudaMemcpy(output_h, output_d, config.numel*sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(output_d);
            cudaFree(input_d);
        }
        else if (config.precision == mxDOUBLE_CLASS) {
            plhs[0] = mxCreateNumericArray(config.ndim, config.dims, config.precision, mxREAL);
            double* output_h = static_cast<double*>(mxGetData(plhs[0]));
            double* input_h = static_cast<double*>(mxGetData(prhs[0]));
            double* output_d;
            double* input_d;

            cudaMalloc((void**)&output_d, config.numel*sizeof(double));
            cudaMalloc((void**)&input_d, config.numel*sizeof(double));
            cudaMemcpy(input_d, input_h, config.numel*sizeof(double), cudaMemcpyHostToDevice);

            auto* cudct = static_cast<transformer<double>*>(config.cudct);
            if (dir == Direction::CUDCT_FORWARD)
                cudct->dct(output_d, input_d);
            else if (dir == Direction::CUDCT_INVERSE)
                cudct->idct(output_d, input_d);

            cudaMemcpy(output_h, output_d, config.numel*sizeof(double), cudaMemcpyDeviceToHost);

            cudaFree(output_d);
            cudaFree(input_d);
        }          
    }
    else if (nlhs == 0 && nrhs == 0 && configured) {
        if (config.precision == mxSINGLE_CLASS)
            delete static_cast<transformer<float>*>(config.cudct);
        else if (config.precision == mxDOUBLE_CLASS)
            delete static_cast<transformer<double>*>(config.cudct);
        config.cudct = nullptr;

        delete[] config.dims;
        configured = false;
    }

    else {
        mexErrMsgTxt("Invalid number of input and output variables given");
    }
}
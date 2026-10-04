## A CUDA implementation of the nD-Discrete Cosine Transform

This repository provides a working implementation to calculate the N-dimensional discrete cosine transform (DCT) with CUDA. It benefits from the relation between the discrete Fourier Transform and DCT coefficients, meaning that this implementation benefits from the FFT methods of the cuFFT library. The appropriate operations relating the FFT and DCT coefficients had already been defined, such as in [the MIRT_DCTN Matlab project](https://www.mathworks.com/matlabcentral/fileexchange/24050-multidimensional-discrete-cosine-transform-dct/files/mirt_idctn.m), which was taken as guideline following the implementation.

The interface is defined with the ```cudct``` class in ```./include/cudct.cuh``` with implementation given in ```./src/cudct.cu```. The class is templated to provide a working implementation for single and double precision arrays. The class contains two public methods, to calculate the forward and inverse DCT.

### Conventions

1. Arrays are assumed to follow column-major ordering (following Matlab/GNU Octave conventions).
2. With the forward DCT, we mean the Type-II DCT. Similarly, with the inverse DCT, we mean the Type-III DCT.
3. This project is compared with the [FFTW](https://fftw.org/) implementation of the DCT.
4. The FFTW library uses [different normalization conventions](https://fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html).


### Project compilation with CMake

TL;DR

```
cmake -S . -B build --config Release
cmake --build build -j
```

With CMake, the cudct class is compiled into a shared library, which is subsequently linked to the benchmark files  in the ```./src``` folder to create executables. Additionally, CMake looks for a Matlab\GNU Octave installation. If installed, it compiles the wrapper functions into Matlab EXecutable (MEX) functions to interact with workspace variables. See also the ```./tests/matlab``` folder for examples. 

A benchmark executable and Matlab wrapper for the FFTW implementation of the DCT is compiled if FFTW can be found in your machine. On Windows, CMake checks the ```fftw``` folder in the working directory. Create this folder yourself, and put the [precompiled .dll files](https://fftw.org/install/windows.html), and follow the instruction to create the .lib files.

## The wrapper functions (for Matlab/GNU Octave)

Originally, this project was initiated to make a CUDA translation of the MIRT_DCTN project in Matlab. In the ```.\src``` are several ```wrapper_*.cpp/cu``` files, which can be used to process workspace variables with the FFTW/CUDA implementations. 

1. One wrapper for the FFTW implementation
2. One wrapper for the implementation via the cudct class
3. A 'static' implementation, which creates a static cudct object, which can be used as long as the dimensions and precision of the input arrays does not change. This reduces computation overhead of initializing the cudct class (which has to allocate memory on the device).
4. A 'gpuArray' implementation, which uses Matlab's gpuArray type as input and output argument. This reduces host-device memory transfer overhead. This wrapper requires the ```gpu/mxGPUArray.h``` header, which is exclusive (to my knowledge) to Matlab (not GNU Octave). In ```./mex/matlab/CMakeLists.txt```, one might have to change the path ```${MATLAB_ROOT_DIR}/toolbox/distcomp/gpu/extern/include``` to meet with your location of this header file.

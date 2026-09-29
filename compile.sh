# FFTW implementation
mkoctfile --mex -O3 -lfftw3 -lfftw3f -o ./mex/octave_fftw.mex src/wrapper_fftw.cpp

# Compile CUDA DCT implementation into object code
nvcc -c -g -O3 -I. -Xcompiler -fPIC src/transformer.cu -o ./mex/transformer.o 

# Compile CUDA DCT into shared library
nvcc -shared -g -O3 -I. -Xcompiler -fPIC ./mex/transformer.o -o ./mex/libtransformer.so -lcufft -lcudart

# Compile wrapper and link against CUDA DCT shared library
mkoctfile --mex -g -O3 src/wrapper_cudct.cpp -o ./mex/octave_cudct.mex -Lmex -ltransformer -Wl,-rpath=mex
mkoctfile --mex -g -O3 src/wrapper_cudct_static.cpp -o ./mex/octave_cudct_static.mex -Lmex -ltransformer -Wl,-rpath=mex

rm ./mex/*.o

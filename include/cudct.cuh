#ifndef _CUDCT_CUH_
#define _CUDCT_CUH_

#include <cufft.h>
#include <vector>

namespace cudct {
    class cudctHandle1D {
        private:
            int _n;
            cufftDoubleComplex* _weights;
            cufftHandle* _plan;
            int* _idcs;

            void set_weights();
            void set_idcs();

        public:
            cudctHandle1D(int n);
            ~cudctHandle1D();

            void exec(cufftReal* idata, cufftReal* odata, int direction);
    };

    class cudctHandle {
        private:
            int _rank;
            std::vector<cudctHandle1D> _handles;

        public:
            cudctHandle(int nx);
            cudctHandle(int nx, int ny);
            cudctHandle(int nx, int ny, int nz);
            cudctHandle(int rank, int* n);
            ~cudctHandle();

            void exec(cufftReal* idata, cufftReal* odata, int direction);
    };
}

#endif
#include <reconstruction_adapt_cuda/angular_continuous_assign2_gpu.h>

int main(int argc, char **argv) {
    ProgCudaAngularContinuousAssign2 program;
    program.read(argc, argv);
    return program.tryRun();
}

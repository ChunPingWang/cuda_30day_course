#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    printf("========================================\n");
    printf("    CUDA Device Information Query\n");
    printf("========================================\n\n");

    // Get CUDA device count
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    // If no CUDA device
    if (deviceCount == 0) {
        printf("Error: No CUDA-capable device found!\n");
        return 1;
    }

    // Iterate through each CUDA device
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("----------------------------------------\n");

        // Compute capability
        printf("  Compute Capability: %d.%d\n",
               deviceProp.major, deviceProp.minor);

        // CUDA cores (simplified calculation)
        int cores = 0;
        int mp = deviceProp.multiProcessorCount;

        // Estimate cores based on compute capability
        if (deviceProp.major == 8) {
            cores = mp * 128; // Ampere architecture
        } else if (deviceProp.major == 7) {
            cores = mp * 64;  // Turing/Volta architecture
        } else if (deviceProp.major == 6) {
            cores = mp * 64;  // Pascal architecture
        } else if (deviceProp.major == 9) {
            cores = mp * 128; // Hopper architecture
        }

        printf("  Streaming Multiprocessors (SM): %d\n", mp);
        printf("  Estimated CUDA Cores: %d\n", cores);

        // Memory info
        printf("  Global Memory: %.2f GB\n",
               deviceProp.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Shared Memory per Block: %.2f KB\n",
               deviceProp.sharedMemPerBlock / 1024.0);
        printf("  Constant Memory: %.2f KB\n",
               deviceProp.totalConstMem / 1024.0);

        // Thread info
        printf("  Max Threads per Block: %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Max Threads per SM: %d\n",
               deviceProp.maxThreadsPerMultiProcessor);

        // Block dimensions
        printf("  Max Block Dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);

        // Grid dimensions
        printf("  Max Grid Dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);

        // Clock rate
        printf("  Clock Rate: %.2f GHz\n",
               deviceProp.clockRate / 1e6);

        // Warp size
        printf("  Warp Size: %d\n", deviceProp.warpSize);

        printf("\n");
    }

    printf("========================================\n");
    printf("CUDA Environment Verification Complete!\n");
    printf("========================================\n");

    return 0;
}

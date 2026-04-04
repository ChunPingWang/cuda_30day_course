#include <stdio.h>
#include <stdlib.h>

/**
 * CUDA kernel function: Vector Addition
 * Each thread computes one element
 */
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    // Calculate global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure we don't exceed array bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    CUDA Vector Addition Demo\n");
    printf("========================================\n\n");

    // Set array size
    const int n = 10;
    size_t bytes = n * sizeof(int);

    // ========== Step 1: Allocate host memory and initialize ==========
    printf("Step 1: Prepare data on host\n");

    int *h_A = (int*)malloc(bytes);  // Host array A
    int *h_B = (int*)malloc(bytes);  // Host array B
    int *h_C = (int*)malloc(bytes);  // Host array C (for results)

    // Initialize data
    printf("A = [ ");
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        printf("%d ", h_A[i]);
    }
    printf("]\n");

    printf("B = [ ");
    for (int i = 0; i < n; i++) {
        h_B[i] = i * 2;
        printf("%d ", h_B[i]);
    }
    printf("]\n\n");

    // ========== Step 2: Allocate device memory ==========
    printf("Step 2: Allocate memory on GPU\n");

    int *d_A, *d_B, *d_C;  // Device pointers
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    printf("Allocated %zu bytes x 3 = %zu bytes on GPU\n\n", bytes, bytes * 3);

    // ========== Step 3: Copy data from host to device ==========
    printf("Step 3: Copy data from CPU to GPU\n");
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    printf("Data transfer complete!\n\n");

    // ========== Step 4: Launch kernel function ==========
    printf("Step 4: Launch GPU kernel function\n");

    int threadsPerBlock = 4;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("Config: <<<{%d}, {%d}>>>\n", blocks, threadsPerBlock);
    printf("Total threads: %d\n\n", blocks * threadsPerBlock);

    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // ========== Step 5: Copy results from device to host ==========
    printf("Step 5: Copy results from GPU to CPU\n");
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("Results transfer complete!\n\n");

    // ========== Step 6: Display results ==========
    printf("Step 6: Display results\n");
    printf("C = A + B = [ ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_C[i]);
    }
    printf("]\n\n");

    // Verify results
    printf("Verifying results...\n");
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf(" Error: C[%d] = %d, expected %d\n", i, h_C[i], h_A[i] + h_B[i]);
            correct = false;
        }
    }
    if (correct) {
        printf(" All results correct!\n\n");
    }

    // ========== Step 7: Free memory ==========
    printf("Step 7: Free memory\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    printf("Memory freed\n");

    printf("\n========================================\n");
    printf(" Program execution complete!\n");
    printf("========================================\n");

    return 0;
}
